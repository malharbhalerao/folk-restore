"""eyeball the collected images fast, and pick the genuinely damaged ones for the real-image demo.

    python scripts/data/contact_sheet.py                        # sheets/sheet_000.jpg ... (30 thumbnails each, numbered)
    python scripts/data/contact_sheet.py --prefix met_          # only one source (museum items are older -> more likely damaged)
    python scripts/data/contact_sheet.py --copy 12 40 77        # MOVE those numbered images to data/real_damaged/

Do this BEFORE step 1: damaged paintings must not stay in data/raw, otherwise they are treated as clean training images.
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from folk_restore import list_images


def thumb(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    canvas = np.full((size, size, 3), 40, np.uint8)
    if img is None:
        return canvas
    h, w = img.shape[:2]
    f = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def label(img, text):
    for col, th in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.putText(img, text, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="sheets")
    ap.add_argument("--style", default=None)
    ap.add_argument("--prefix", default="", help="only files whose name starts with this, e.g. met_ cma_ aic_ commons_")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--thumb", type=int, default=220)
    ap.add_argument("--copy", type=int, nargs="*", default=None, help="numbers to move into --dest")
    ap.add_argument("--dest", default="data/real_damaged")
    a = ap.parse_args()
    out, index_path = Path(a.out), Path(a.out) / "index.csv"

    if a.copy is not None:
        rows = {int(r["id"]): r["path"] for r in csv.DictReader(open(index_path))}
        for i in a.copy:
            src = Path(rows[i])
            if not src.exists():
                print(f"  #{i}: already moved or missing ({src})")
                continue
            dst = Path(a.dest) / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dst)
            print(f"  #{i}: {src.name} -> {dst}")
        return

    files = [p for p in list_images(a.raw) if p.name.startswith(a.prefix) and (not a.style or p.parent.name == a.style)]
    if not files:
        raise SystemExit("no images found")
    out.mkdir(parents=True, exist_ok=True)
    per = a.cols * a.rows
    with open(index_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "path"])
        for i, p in enumerate(files):
            w.writerow([i, str(p)])
    for s in range(0, len(files), per):
        canvas = np.full((a.rows * a.thumb, a.cols * a.thumb, 3), 40, np.uint8)
        for k, p in enumerate(files[s:s + per]):
            t = thumb(p, a.thumb)
            label(t, str(s + k))
            r, c = divmod(k, a.cols)
            canvas[r * a.thumb:(r + 1) * a.thumb, c * a.thumb:(c + 1) * a.thumb] = t
        cv2.imwrite(str(out / f"sheet_{s // per:03d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(files)} images -> {(len(files) + per - 1) // per} sheets in {out}/ (numbers match {index_path})")


if __name__ == "__main__":
    main()
