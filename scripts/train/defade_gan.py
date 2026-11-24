"""Pix2Pix colour de-fader with protected-margin references.

Improves on step 8 (U-Net, L1 only, no references) in the two ways the brief asks for:
  1. adversarial training: a conditional PatchGAN critic (Pix2Pix) next to the L1 loss;
  2. margin references: the generator is also given the colour statistics of an un-faded strip (the part of a painting that
     was protected by the frame), so it can learn THIS painting's palette instead of an average one.
Training pairs come from the same fading simulator as step 8. 30% of samples carry no reference (flag = 0) so the model also
works when no margin is available.

    python scripts/train/defade_gan.py                      # ~15-25 min on a T4 (estimate, not measured)
    python scripts/train/defade_gan.py --lambda_gan 0       # ablation: same model, no adversarial loss
    python scripts/train/defade_gan.py --epochs 1 --samples_per_epoch 8 --bs 4 --workers 0 --no_pretrained --n_val 4 --test_n 4   # smoke test
Outputs: runs/defade_gan/best.pt, log.csv; results/defade_gan.md, results/defade_gan_samples/
"""
import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from folk_restore import imread_rgb, imwrite_rgb, load_defade, defade_image
from folk_restore.colour import DEFADE_NORM, build_g, d_input, defade_ref_image, g_forward, lab_of, margin_strip, ref_stats, ref_vector
from folk_restore.degrade import apply_fade, make_fade_strength
from folk_restore.gan import PatchD
from folk_restore.metrics import de00


class FadeRefDS(Dataset):
    def __init__(self, data, crop, ref_prob):
        self.data, self.crop, self.ref_prob = Path(data), crop, ref_prob
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
        faded = clean if random.random() < 0.15 else apply_fade(clean, make_fade_strength(c, c, np.random.default_rng()))
        if random.random() < self.ref_prob:                      # a protected margin is available: it is UN-faded
            ref = ref_vector(ref_stats(margin_strip(clean, random.randint(0, 3), random.randint(24, 48))), True)
        else:                                                    # no margin: the model only sees the faded tile's own stats
            ref = ref_vector(ref_stats(faded), False)
        lab = lambda im: torch.from_numpy(lab_of(im).transpose(2, 0, 1).copy())  # noqa: E731
        return lab(faded), lab(clean), torch.from_numpy(ref)


def evaluate(model, data, split, n, device):
    """mean dE00 of: faded input, de-faded without margin, de-faded with a margin (edge rotates per tile)."""
    folder = Path(data) / "synthetic" / split
    idx = list(csv.DictReader(open(folder / "index.csv")))[:n]
    base, noref, withref = [], [], []
    for k, r in enumerate(idx):
        clean = imread_rgb(Path(data) / r["path"])
        faded = imread_rgb(folder / f"{r['tile_id']}_faded.png")
        base.append(de00(faded, clean))
        noref.append(de00(defade_ref_image(model, faded, device, None), clean))
        withref.append(de00(defade_ref_image(model, faded, device, ref_stats(margin_strip(clean, k))), clean))
    return float(np.mean(base)), float(np.mean(noref)), float(np.mean(withref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="defade_gan")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples_per_epoch", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--lambda_l1", type=float, default=100.0, help="Pix2Pix default")
    ap.add_argument("--lambda_gan", type=float, default=1.0, help="0 = ablation without the adversarial loss")
    ap.add_argument("--ref_prob", type=float, default=0.7, help="fraction of training samples that get a margin reference")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--test_n", type=int, default=100)
    ap.add_argument("--l1_model", default="runs/defade/best.pt", help="step-8 model for the comparison row (skipped if missing)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no_pretrained", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path("runs") / a.name
    run.mkdir(parents=True, exist_ok=True)
    ds = FadeRefDS(a.data, a.crop, a.ref_prob)
    dl = DataLoader(ds, batch_size=a.bs, sampler=RandomSampler(ds, replacement=True, num_samples=a.samples_per_epoch),
                    num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    G = build_g(pretrained=not a.no_pretrained).to(device)
    D = PatchD(in_ch=6).to(device)
    opt_g = torch.optim.AdamW(G.parameters(), lr=a.lr, weight_decay=1e-4)
    opt_d = torch.optim.Adam(D.parameters(), lr=a.lr_d, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=a.epochs)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    norm = torch.tensor(DEFADE_NORM, device=device).view(1, 3, 1, 1)

    logf = open(run / "log.csv", "w", newline="")
    wr = csv.writer(logf)
    wr.writerow(["epoch", "l1", "gan_g", "d_loss", "val_de00_faded", "val_de00_noref", "val_de00_ref", "seconds"])
    best = 1e9
    for ep in range(1, a.epochs + 1):
        G.train()
        t0, acc, n = time.time(), np.zeros(3), 0
        for faded, clean, ref in dl:
            faded, clean, ref = faded.to(device), clean.to(device), ref.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                delta = g_forward(G, faded, ref)
            pred = faded + delta.float()
            l1 = ((pred - clean).abs() / norm).mean()
            loss_gan = torch.zeros((), device=device)
            if a.lambda_gan > 0:
                loss_gan = ((D(d_input(faded, pred)) - 1) ** 2).mean()          # LSGAN, as in Pix2Pix
            loss_g = a.lambda_gan * loss_gan + a.lambda_l1 * l1
            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()
            loss_d = torch.zeros((), device=device)
            if a.lambda_gan > 0:
                loss_d = 0.5 * (((D(d_input(faded, clean)) - 1) ** 2).mean() + (D(d_input(faded, pred.detach())) ** 2).mean())
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
            acc += [l1.item(), loss_gan.item(), loss_d.item()]
            n += 1
        sched.step()
        acc /= max(n, 1)
        G.eval()
        b, nr, wr_ = evaluate(G, a.data, "val", a.n_val, device)
        wr.writerow([ep, *[round(float(v), 5) for v in acc], round(b, 3), round(nr, 3), round(wr_, 3), round(time.time() - t0)])
        logf.flush()
        print(f"epoch {ep:2d}/{a.epochs}  L1 {acc[0]:.4f}  GAN {acc[1]:.3f}  D {acc[2]:.3f}  val dE00: faded {b:.2f} -> "
              f"no-margin {nr:.2f} | with-margin {wr_:.2f}  ({time.time() - t0:.0f}s)")
        ck = {"model": G.state_dict(), "encoder": "resnet34", "epoch": ep, "val_de00_noref": nr, "val_de00_ref": wr_}
        torch.save(ck, run / "last.pt")
        score = 0.5 * (nr + wr_)
        if score < best:
            best = score
            torch.save(ck, run / "best.pt")

    # ---- final numbers on the held-out synthetic TEST set
    G.load_state_dict(torch.load(run / "best.pt", map_location=device)["model"])
    G.eval()
    l1m = load_defade(a.l1_model, device) if Path(a.l1_model).exists() else None
    out = Path(a.out)
    (out / "defade_gan_samples").mkdir(parents=True, exist_ok=True)
    folder = Path(a.data) / "synthetic" / "test"
    idx = list(csv.DictReader(open(folder / "index.csv")))[:a.test_n]
    res = {"faded input (no correction)": [], "step 8: U-Net, L1 only, no reference": [],
           "Pix2Pix, no margin reference": [], "Pix2Pix + margin reference": [],
           "healthy tile after Pix2Pix, no margin (should stay ~0)": [],
           "healthy tile after Pix2Pix + its own margin (should stay ~0)": []}
    for k, r in enumerate(idx):
        clean = imread_rgb(Path(a.data) / r["path"])
        faded = imread_rgb(folder / f"{r['tile_id']}_faded.png")
        stats = ref_stats(margin_strip(clean, k))
        d_ref = defade_ref_image(G, faded, device, stats)
        res["faded input (no correction)"].append(de00(faded, clean))
        if l1m is not None:
            res["step 8: U-Net, L1 only, no reference"].append(de00(defade_image(l1m, faded, device, 1.0), clean))
        res["Pix2Pix, no margin reference"].append(de00(defade_ref_image(G, faded, device, None), clean))
        res["Pix2Pix + margin reference"].append(de00(d_ref, clean))
        if k < 30:
            res["healthy tile after Pix2Pix, no margin (should stay ~0)"].append(de00(defade_ref_image(G, clean, device, None), clean))
            res["healthy tile after Pix2Pix + its own margin (should stay ~0)"].append(de00(defade_ref_image(G, clean, device, stats), clean))
        if k < 8:
            s = out / "defade_gan_samples"
            imwrite_rgb(s / f"{r['tile_id']}_cleanref.png", clean)
            imwrite_rgb(s / f"{r['tile_id']}_faded.png", faded)
            imwrite_rgb(s / f"{r['tile_id']}_defade.png", d_ref)
    lines = ["| Method | mean dE00 vs clean (lower = better) |", "| --- | --- |"]
    lines += [f"| {k} | {np.mean(v):.2f} |" for k, v in res.items() if v]
    (out / "defade_gan.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nmodel -> {run / 'best.pt'} | samples -> {out / 'defade_gan_samples'}")


if __name__ == "__main__":
    main()
