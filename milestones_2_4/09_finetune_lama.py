"""Step 9 (Milestone 2): fine-tune pretrained Big-LaMa on OUR tiles, with synthetic damage masks.

Why fine-tune instead of the zero-shot model of the progress report:
  * the brief asks for a *trained* inpainting model;
  * our damage (hairline cracks, flaking, stains on folk-art colour fields and line work) differs from LaMa's training
    data (Places2 photos, random masks).

Training data is generated on the fly: tile -> degrade() (cracks / losses / stains, optionally with unlabelled design-line
decoys) -> mask = dilated union of the true damage masks, plus extra large blobs half the time. Target = the clean tile.
Loss: masked L1 + VGG perceptual + PatchGAN hinge. The PatchGAN critic is trained from scratch alongside and is saved,
because Milestone 4 reuses it as the discriminator-based uncertainty signal.

    python 09_finetune_lama.py                                   # ~30-40 min on a T4 (estimate, not measured)
    python 09_finetune_lama.py --epochs 1 --samples_per_epoch 4 --bs 1 --crop 128 --n_val 2 --no_perc   # smoke test
Outputs: runs/lama_ft/best.pt (TorchScript, loadable by LamaJit), disc.pt, log.csv
"""
import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from common import CLASSES, dilate_mask, imread_rgb
from degrade import degrade
from inpaint_lib import PatchD, VGGPerceptual, LamaJit, load_lama_jit, random_blob_mask, save_disc


class InpaintDS(Dataset):
    def __init__(self, data, crop, decoy_prob, blob_prob):
        self.data, self.crop, self.decoy_prob, self.blob_prob = Path(data), crop, decoy_prob, blob_prob
        self.rows = [r for r in csv.DictReader(open(self.data / "manifest.csv")) if r["split"] == "train"]
        if not self.rows:
            raise SystemExit("No training tiles in manifest.csv")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        clean = imread_rgb(self.data / self.rows[i]["path"])
        H, W = clean.shape[:2]
        c = self.crop
        y, x = random.randint(0, H - c), random.randint(0, W - c)
        clean = np.ascontiguousarray(np.rot90(clean[y:y + c, x:x + c], random.randint(0, 3)))
        if random.random() < 0.5:
            clean = np.ascontiguousarray(clean[:, ::-1])
        rng = np.random.default_rng()
        d = degrade(clean, rng, fade_prob=0.0, decoy_prob=self.decoy_prob)
        union = np.zeros(clean.shape[:2], np.uint8)
        for k in CLASSES:
            union |= d["masks"][k]
        mask = dilate_mask(union, 5) // 255
        if random.random() < self.blob_prob:
            mask = np.maximum(mask, random_blob_mask(c, c, rng))
        if mask.sum() == 0:                                  # never train on an empty mask
            mask = random_blob_mask(c, c, rng, 1)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
        img = t(d["image_struct"].transpose(2, 0, 1)).float() / 255.0
        tgt = t(d["clean"].transpose(2, 0, 1)).float() / 255.0
        return img, tgt, t(mask[None]).float()


def val_psnr(model, folder, data, n, device):
    """PSNR on the damaged region of the fixed synthetic val tiles (oracle masks): the same protocol as step 4."""
    rows = list(csv.DictReader(open(Path(folder) / "index.csv")))[:n]
    model.eval()
    vals = []
    with torch.no_grad():
        for r in rows:
            tid = r["tile_id"]
            struct = imread_rgb(Path(folder) / f"{tid}_struct.png")
            clean = imread_rgb(Path(data) / r["path"])
            gt = (imread_rgb(Path(folder) / f"{tid}_mask.png") > 127).any(axis=2).astype(np.uint8)
            m = dilate_mask(gt, 5)
            H, W = struct.shape[:2]
            ph, pw = (-H) % 8, (-W) % 8
            x = torch.from_numpy(struct.astype(np.float32).transpose(2, 0, 1) / 255.0)[None].to(device)
            mk = torch.from_numpy((m > 0).astype(np.float32))[None, None].to(device)
            if ph or pw:
                x, mk = F.pad(x, (0, pw, 0, ph), mode="reflect"), F.pad(mk, (0, pw, 0, ph), mode="reflect")
            out = model(x, mk)[0].permute(1, 2, 0).float().cpu().numpy()[:H, :W]
            out = np.clip(out * 255, 0, 255).astype(np.uint8)
            reg = m > 0
            if reg.sum():
                mse = np.mean((out[reg].astype(np.float64) - clean[reg].astype(np.float64)) ** 2)
                vals.append(99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse))
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="lama_ft")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--samples_per_epoch", type=int, default=800)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr_g", type=float, default=2e-5, help="small: we are fine-tuning, not retraining")
    ap.add_argument("--lr_d", type=float, default=1e-4)
    ap.add_argument("--w_l1", type=float, default=10.0)
    ap.add_argument("--w_perc", type=float, default=1.0)
    ap.add_argument("--w_adv", type=float, default=0.05)
    ap.add_argument("--adv_warmup", type=float, default=0.25, help="fraction of steps over which the adversarial weight ramps up")
    ap.add_argument("--decoy_prob", type=float, default=0.3)
    ap.add_argument("--blob_prob", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--no_perc", action="store_true", help="skip the VGG loss (only for offline smoke tests)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path("runs") / a.name
    run.mkdir(parents=True, exist_ok=True)
    G = load_lama_jit(None, device)
    G.train()
    D = PatchD().to(device)
    perc = None
    if not a.no_perc:
        perc = VGGPerceptual().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=a.lr_g, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=a.lr_d, betas=(0.5, 0.9))

    ds = InpaintDS(a.data, a.crop, a.decoy_prob, a.blob_prob)
    dl = DataLoader(ds, batch_size=a.bs, sampler=RandomSampler(ds, replacement=True, num_samples=a.samples_per_epoch),
                    num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    val_folder = Path(a.data) / "synthetic" / "val"
    total_steps, step = a.epochs * (a.samples_per_epoch // a.bs), 0

    base = val_psnr(G, val_folder, a.data, a.n_val, device)
    print(f"zero-shot Big-LaMa, val PSNR (damaged region, oracle masks): {base:.2f} dB   <- baseline to beat")
    logf = open(run / "log.csv", "w", newline="")
    wr = csv.writer(logf)
    wr.writerow(["epoch", "l1", "perceptual", "adv_g", "d_loss", "val_psnr_region", "seconds"])
    wr.writerow([0, "", "", "", "", base, 0])
    best = base
    torch.jit.save(G, run / "best.pt")                      # epoch 0 = the pretrained model, so best.pt always exists
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        acc = np.zeros(4)
        n = 0
        for img, tgt, mask in dl:
            img, tgt, mask = img.to(device), tgt.to(device), mask.to(device)
            w_adv = a.w_adv * min(1.0, step / max(1, a.adv_warmup * total_steps))
            pred = G(img, mask)                              # LaMa returns the composite: hole <- generated, rest <- input
            l1 = (torch.abs(pred - tgt) * mask).sum() / (mask.sum() * 3 + 1e-6)
            lp = perc(pred, tgt) if perc is not None else torch.zeros((), device=device)
            adv = -D(pred).mean()
            loss_g = a.w_l1 * l1 + a.w_perc * lp + w_adv * adv
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g.step()
            # critic: hinge loss, real = clean tile, fake = LaMa composite
            d_real, d_fake = D(tgt), D(pred.detach())
            loss_d = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            acc += [l1.item(), lp.item(), adv.item(), loss_d.item()]
            n += 1
            step += 1
        acc /= max(n, 1)
        psnr = val_psnr(G, val_folder, a.data, a.n_val, device)
        wr.writerow([ep, *[round(float(v), 5) for v in acc], round(psnr, 3), round(time.time() - t0)])
        logf.flush()
        print(f"epoch {ep:2d}/{a.epochs}  L1 {acc[0]:.4f}  perc {acc[1]:.3f}  adv {acc[2]:+.3f}  D {acc[3]:.3f}  "
              f"val PSNR {psnr:.2f} dB (zero-shot {base:.2f})  ({time.time() - t0:.0f}s)")
        torch.jit.save(G, run / "last.pt")
        save_disc(run / "disc_last.pt", D)
        if psnr > best:
            best = psnr
            torch.jit.save(G, run / "best.pt")
            save_disc(run / "disc.pt", D)
    if not (run / "disc.pt").exists():                       # fine-tuning never beat zero-shot: keep the critic anyway
        save_disc(run / "disc.pt", D)
    print(f"\nbest val PSNR {best:.2f} dB (zero-shot {base:.2f}) -> {run / 'best.pt'}")
    if best <= base:
        print("[note] fine-tuning did not beat the zero-shot model on validation; best.pt is the pretrained model. "
              "Report that honestly, and try more epochs / lower --lr_g / --w_adv 0.")


if __name__ == "__main__":
    main()
