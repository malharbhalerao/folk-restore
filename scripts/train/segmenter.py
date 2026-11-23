"""train the damage segmenter (U-Net, ResNet-34 encoder) on on-the-fly synthetic damage.

Examples:
    python scripts/train/segmenter.py                       # main model (RGB + CLAHE-L + black-hat input)
    python scripts/train/segmenter.py --mode rgb --name unet_rgb    # ablation: plain RGB input
    python scripts/train/segmenter.py --epochs 1 --samples_per_epoch 64 --no_pretrained   # smoke test
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

from folk_restore import CLASSES, build_model, features, imread_rgb
from folk_restore.degrade import degrade, degrade_v2


class TrainDS(Dataset):
    def __init__(self, data, mode, crop, aug="v1"):
        self.data, self.mode, self.crop, self.aug = Path(data), mode, crop, aug
        self.rows = [r for r in csv.DictReader(open(self.data / "manifest.csv")) if r["split"] == "train"]
        if not self.rows:
            raise SystemExit("No training tiles in manifest.csv")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        clean = imread_rgb(self.data / self.rows[i]["path"])
        if self.aug == "v2":    # scale augmentation: real images are run at a different pixel scale than the tiles
            f = random.uniform(0.5, 1.2)
            clean = cv2.resize(clean, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        H, W = clean.shape[:2]
        c = self.crop
        if min(H, W) < c:
            clean = cv2.resize(clean, (max(W, c), max(H, c)), interpolation=cv2.INTER_CUBIC)
            H, W = clean.shape[:2]
        y, x = random.randint(0, H - c), random.randint(0, W - c)
        clean = clean[y:y + c, x:x + c]
        clean = np.rot90(clean, random.randint(0, 3))
        if random.random() < 0.5:
            clean = clean[:, ::-1]
        rng = np.random.default_rng()
        clean = np.ascontiguousarray(clean)
        d = degrade_v2(clean, rng) if self.aug == "v2" else degrade(clean, rng, fade_prob=0.6)
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
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--aug", choices=["v1", "v2"], default="v1", help="v2 = distractor strokes, clean negatives, photometric + scale aug")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    import segmentation_models_pytorch as smp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={a.mode}")
    run = Path("runs") / a.name
    run.mkdir(parents=True, exist_ok=True)

    train_ds = TrainDS(a.data, a.mode, a.crop, a.aug)
    sampler = RandomSampler(train_ds, replacement=True, num_samples=a.samples_per_epoch)
    train_dl = DataLoader(train_ds, batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                          drop_last=True, persistent_workers=a.workers > 0)
    val_dl = DataLoader(FixedDS(Path(a.data) / "synthetic" / "val", a.mode, a.n_val), batch_size=8, num_workers=0)

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
