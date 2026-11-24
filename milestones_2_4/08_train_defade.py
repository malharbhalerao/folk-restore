"""Step 8 (Milestone 3): learned colour de-fader.

A U-Net looks at a faded tile and predicts a LAB colour correction (a residual added to the input), trained on pairs made
by the fading simulator (clean tile -> faded tile). ~15% of training samples are NOT faded, so the model learns to leave
healthy colours alone. Evaluated with CIEDE2000 on the fixed synthetic test set, next to "faded input" (do nothing).

    python 08_train_defade.py                  # ~10-15 min on a T4
    python 08_train_defade.py --epochs 1 --samples_per_epoch 8 --bs 4 --workers 0 --no_pretrained --n_val 4 --test_n 4   # smoke test
"""
import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab
from torch.utils.data import DataLoader, Dataset, RandomSampler

from common import (build_defade_model, defade_forward, defade_image, imread_rgb, imwrite_rgb)
from degrade import apply_fade, make_fade_strength


class FadeDS(Dataset):
    def __init__(self, data, crop):
        self.data, self.crop = Path(data), crop
        self.rows = [r for r in csv.DictReader(open(self.data / "manifest.csv")) if r["split"] == "train"]

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
        if random.random() < 0.15:
            faded = clean                                   # healthy tile: the right answer is "change nothing"
        else:
            faded = apply_fade(clean, make_fade_strength(c, c, np.random.default_rng()))
        lab = lambda im: torch.from_numpy(cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).transpose(2, 0, 1).copy())  # noqa: E731
        return lab(faded), lab(clean)


def de00(a, b):
    return float(deltaE_ciede2000(rgb2lab(a), rgb2lab(b)).mean())


def evaluate(model, data, split, n, device, strength=1.0):
    folder = Path(data) / "synthetic" / split
    idx = list(csv.DictReader(open(folder / "index.csv")))[:n]
    base, out = [], []
    for r in idx:
        clean = imread_rgb(Path(data) / r["path"])
        faded = imread_rgb(folder / f"{r['tile_id']}_faded.png")
        base.append(de00(faded, clean))
        out.append(de00(defade_image(model, faded, device, strength), clean))
    return float(np.mean(base)), float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="defade")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples_per_epoch", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--test_n", type=int, default=100)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no_pretrained", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path("runs") / a.name
    run.mkdir(parents=True, exist_ok=True)
    ds = FadeDS(a.data, a.crop)
    dl = DataLoader(ds, batch_size=a.bs, sampler=RandomSampler(ds, replacement=True, num_samples=a.samples_per_epoch),
                    num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    model = build_defade_model(pretrained=not a.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = 1e9
    logf = open(run / "log.csv", "w", newline="")
    logw = csv.writer(logf)
    logw.writerow(["epoch", "train_loss", "val_de00_faded", "val_de00_defaded", "seconds"])
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for lab_f, lab_c in dl:
            lab_f, lab_c = lab_f.to(device), lab_c.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                delta = defade_forward(model, lab_f)
            loss = (lab_f + delta.float() - lab_c).abs().mean() / 10.0     # L1 in LAB units
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n += 1
        sched.step()
        model.eval()
        base, got = evaluate(model, a.data, "val", a.n_val, device)
        logw.writerow([ep, tot / max(n, 1), base, got, round(time.time() - t0)])
        logf.flush()
        print(f"epoch {ep:2d}/{a.epochs}  loss {tot / max(n, 1):.4f}  val dE00: faded input {base:.2f} -> de-faded {got:.2f}  ({time.time() - t0:.0f}s)")
        ck = {"model": model.state_dict(), "encoder": "resnet34", "epoch": ep, "val_de00": got}
        torch.save(ck, run / "last.pt")
        if got < best:
            best = got
            torch.save(ck, run / "best.pt")

    # ---- final numbers on the held-out synthetic TEST set (+ a sanity check on un-faded tiles)
    ck = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    out = Path(a.out)
    (out / "defade_learned_samples").mkdir(parents=True, exist_ok=True)
    folder = Path(a.data) / "synthetic" / "test"
    idx = list(csv.DictReader(open(folder / "index.csv")))[:a.test_n]
    res = {"faded input (no correction)": [], "learned de-fader, strength 0.7": [], "learned de-fader, strength 1.0": [],
           "healthy tile, untouched (reference)": [], "healthy tile after de-fader (should stay ~0)": []}
    for k, r in enumerate(idx):
        clean = imread_rgb(Path(a.data) / r["path"])
        faded = imread_rgb(folder / f"{r['tile_id']}_faded.png")
        d1 = defade_image(model, faded, device, 1.0)
        res["faded input (no correction)"].append(de00(faded, clean))
        res["learned de-fader, strength 1.0"].append(de00(d1, clean))
        res["learned de-fader, strength 0.7"].append(de00(defade_image(model, faded, device, 0.7), clean))
        if k < 30:
            res["healthy tile, untouched (reference)"].append(0.0)
            res["healthy tile after de-fader (should stay ~0)"].append(de00(defade_image(model, clean, device, 1.0), clean))
        if k < 8:
            s = out / "defade_learned_samples"
            imwrite_rgb(s / f"{r['tile_id']}_cleanref.png", clean)
            imwrite_rgb(s / f"{r['tile_id']}_faded.png", faded)
            imwrite_rgb(s / f"{r['tile_id']}_defade.png", d1)
    lines = ["| Method | mean dE00 vs clean (lower = better) |", "| --- | --- |"]
    lines += [f"| {k} | {np.mean(v):.2f} |" for k, v in res.items() if v]
    (out / "defade_learned.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nmodel -> {run / 'best.pt'} | samples -> {out / 'defade_learned_samples'}")


if __name__ == "__main__":
    main()
