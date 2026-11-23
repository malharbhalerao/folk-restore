"""Step 7: slide-ready panels. Looks for files named <stem>_<suffix>.png in a folder and lines them up.

    python 07_make_figures.py --dir results/samples --out figures/synthetic     # step-4 samples
    python 07_make_figures.py --dir results/real    --out figures/real          # step-5 outputs
    python 07_make_figures.py --dir results/defade_samples --out figures/defade # step-6 samples
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = [("clean", "Clean (ground truth)"), ("cleanref", "Original"), ("faded", "Faded input"),
         ("input", "Input"), ("damaged", "Damaged"), ("mask", "Predicted mask"), ("unc", "Uncertainty"),
         ("telea", "OpenCV Telea"), ("lama", "LaMa"), ("defade", "De-faded"), ("sr", "Super-resolved")]


def load(p):
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/samples")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--stems", nargs="*", default=None)
    a = ap.parse_args()
    d, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    stems = a.stems
    if not stems:
        known = [s for s, _ in ORDER]
        stems = sorted({f.stem.rsplit("_", 1)[0] for f in d.glob("*.png") if f.stem.rsplit("_", 1)[-1] in known})
    for stem in stems[:a.limit]:
        cols = [(s, t) for s, t in ORDER if (d / f"{stem}_{s}.png").exists()]
        if len(cols) < 2:
            continue
        fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.6))
        for ax, (s, t) in zip(axes, cols):
            ax.imshow(load(d / f"{stem}_{s}.png"), cmap="gray" if s == "mask" else None)
            ax.set_title(t, fontsize=11)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=150)
        plt.close(fig)
        print("wrote", out / f"{stem}.png")


if __name__ == "__main__":
    main()
