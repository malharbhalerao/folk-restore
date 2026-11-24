"""Step 3: train the damage segmenter (U-Net, ResNet-34 encoder) on on-the-fly synthetic damage.

Examples:
    python 03_train_unet.py                       # main model (RGB + CLAHE-L + black-hat input)
    python 03_train_unet.py --mode rgb --name unet_rgb    # ablation: plain RGB input
    python 03_train_unet.py --epochs 1 --samples_per_epoch 64 --no_pretrained   # smoke test
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

from common import CLASSES, build_model, features, imread_rgb
from degrade import degrade


class TrainDS(Dataset):
    def __init__(self, data, mode, crop, decoy_prob=0.0, profile='paper', scale_jitter=0.0, photo=0.0):
        self.data, self.mode, self.crop, self.decoy_prob = Path(data), mode, crop, decoy_prob
        self.profile, self.scale_jitter, self.photo = profile, scale_jitter, photo
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
        clean = clean[y:y + c, x:x + c]
        clean = np.rot90(clean, random.randint(0, 3))
        if random.random() < 0.5:
            clean = clean[:, ::-1]
        d = degrade(np.ascontiguousarray(clean), np.random.default_rng(), fade_prob=0.6, decoy_prob=self.decoy_prob,
                    profile=self.profile, scale_jitter=self.scale_jitter, photo=self.photo)
        f = features(d["image"], self.mode).transpose(2, 0, 1)
        t = np.stack([d["masks"][k] for k in CLASSES], 0).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(f)), torch.from_numpy(t)


class FixedDS(Dataset):
    def __init__(self, folder, mode, n):
        folder = Path(folder)
        ids = [r["tile_id"] for r in csv.DictReader(open(folder / "index.csv"))][:n]
        self.folder, self.ids, self.mode = folder, ids, mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        tid = self.ids[i]
        img = imread_rgb(self.folder / f"{tid}_damaged.png")
        m = imread_rgb(self.folder / f"{tid}_mask.png") > 127
        f = features(img, self.mode).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(f)), torch.from_numpy(m.transpose(2, 0, 1).astype(np.float32))


def evaluate(model, loader, device):
    model.eval()
    inter = np.zeros(len(CLASSES))
    p_sum = np.zeros(len(CLASSES))
    g_sum = np.zeros(len(CLASSES))
    ui = up = ug = 0.0
    with torch.no_grad():
        for x, y in loader:
            p = (torch.sigmoid(model(x.to(device)).float()) > 0.5).cpu().numpy()
            y = y.numpy() > 0.5
            inter += (p & y).sum(axis=(0, 2, 3))
            p_sum += p.sum(axis=(0, 2, 3))
            g_sum += y.sum(axis=(0, 2, 3))
            pu, gu = p.any(1), y.any(1)
            ui += (pu & gu).sum()
            up += pu.sum()
            ug += gu.sum()
    eps = 1e-6
    out = {f"dice_{c}": float((2 * inter[i] + eps) / (p_sum[i] + g_sum[i] + eps)) for i, c in enumerate(CLASSES)}
    out["dice_union"] = float((2 * ui + eps) / (up + ug + eps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--mode", choices=["rich", "rgb"], default="rich")
    ap.add_argument("--name", default="unet_rich")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples_per_epoch", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--decoy_prob", type=float, default=0.0,
                    help="fraction of training tiles that get UNLABELLED design lines (hard negatives); try 0.4 for the v2 model")
    ap.add_argument("--profile", choices=["paper", "wall", "mixed"], default="paper",
                    help="damage simulator: paper (original), wall (plaster flaking/salt/streaks), mixed (50/50)")
    ap.add_argument("--scale_jitter", type=float, default=0.0, help="prob. damage is rendered at a random 0.5-2x scale")
    ap.add_argument("--photo", type=float, default=0.0, help="prob. of phone-photo blur/noise/JPEG/uneven light")
    ap.add_argument("--val_dir", default=None, help="fixed validation folder (default data/synthetic/val)")
    ap.add_argument("--no_pretrained", action="store_true")
    a = ap.parse_args()

    import segmentation_models_pytorch as smp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={a.mode}")
    run = Path("runs") / a.name
    run.mkdir(parents=True, exist_ok=True)

    train_ds = TrainDS(a.data, a.mode, a.crop, a.decoy_prob, a.profile, a.scale_jitter, a.photo)
    sampler = RandomSampler(train_ds, replacement=True, num_samples=a.samples_per_epoch)
    train_dl = DataLoader(train_ds, batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                          drop_last=True, persistent_workers=a.workers > 0)
    val_dl = DataLoader(FixedDS(Path(a.val_dir) if a.val_dir else Path(a.data) / "synthetic" / "val", a.mode, a.n_val), batch_size=8, num_workers=0)

    model = build_model(a.mode, a.encoder, pretrained=not a.no_pretrained).to(device)
    dice_loss = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log = open(run / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "train_loss", "dice_crack", "dice_loss", "dice_stain", "dice_union", "seconds"])
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            loss = bce(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n += 1
        sched.step()
        m = evaluate(model, val_dl, device)
        dt = time.time() - t0
        wr.writerow([ep, tot / max(n, 1), m["dice_crack"], m["dice_loss"], m["dice_stain"], m["dice_union"], round(dt)])
        log.flush()
        print(f"epoch {ep:2d}/{a.epochs}  loss {tot / max(n, 1):.4f}  val Dice  crack {m['dice_crack']:.3f}  "
              f"loss {m['dice_loss']:.3f}  stain {m['dice_stain']:.3f}  union {m['dice_union']:.3f}  ({dt:.0f}s)")
        ck = {"model": model.state_dict(), "mode": a.mode, "encoder": a.encoder, "epoch": ep, "val": m}
        torch.save(ck, run / "last.pt")
        if m["dice_union"] > best:
            best = m["dice_union"]
            torch.save(ck, run / "best.pt")
    print(f"\nbest val union Dice = {best:.3f}  ->  {run / 'best.pt'}")


if __name__ == "__main__":
    main()
