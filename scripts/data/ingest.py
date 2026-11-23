"""Step 0b: add images you gathered yourself (a Kaggle input dataset, a folder of manual downloads) to data/raw/.

    # everything in the folder is one style:
    python scripts/data/ingest.py --src /kaggle/input/some-madhubani-dataset --style madhubani --source_name kaggle-xyz --license "see dataset page"
    # style guessed from sub-folder / file names containing madhubani, warli, pattachitra, kalamkari:
    python scripts/data/ingest.py --src /kaggle/input/indian-paintings --auto_style --source_name kaggle-abc

Files are COPIED (the source stays untouched) with collision-proof names, and logged in data/sources.csv.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

from folk_restore import STYLES, append_sources, list_images


def guess_style(path):
    low = str(path).lower().replace("-", "").replace("_", "").replace(" ", "")
    alias = {"madhubani": "madhubani", "mithila": "madhubani", "warli": "warli", "worli": "warli",
             "pattachitra": "pattachitra", "patachitra": "pattachitra", "kalamkari": "kalamkari"}
    for k, v in alias.items():
        if k in low:
            return v
    return "unsorted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--style", choices=STYLES, default=None)
    ap.add_argument("--auto_style", action="store_true", help="guess style from the path")
    ap.add_argument("--out", default="data")
    ap.add_argument("--source_name", default="manual")
    ap.add_argument("--license", default="unknown - check source", dest="license_")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if not a.style and not a.auto_style:
        raise SystemExit("Give --style <name> or --auto_style")

    src, out = Path(a.src), Path(a.out)
    files = list_images(src)
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"No images found under {src}")
    rows, counts = [], {}
    for p in files:
        style = a.style or guess_style(p.relative_to(src))
        tag = hashlib.md5(str(p).encode()).hexdigest()[:6]
        name = f"{a.source_name}_{p.stem[:40]}_{tag}{p.suffix.lower()}"
        dest = out / "raw" / style / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(p, dest)
        counts[style] = counts.get(style, 0) + 1
        rows.append(dict(file=f"{style}/{name}", style=style, source=a.source_name, url=str(p),
                         license=a.license_, title=p.stem))
    append_sources(out / "sources.csv", rows)
    print(f"copied {len(files)} images:", counts)
    if "unsorted" in counts:
        print("note: 'unsorted' images had no style keyword in their path. Move them into the right data/raw/<style>/ folder or keep as is "
              "(they are still used for training; the style label only affects the stratified split).")


if __name__ == "__main__":
    main()
