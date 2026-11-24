"""Score the human audit: detector precision and blind A/B preference, averaged over raters.

    python scripts/eval/audit_score.py --dir audit     # expects audit/labels_<rater>.csv and audit/ab_<rater>.csv
"""
import argparse
import csv
import glob
from collections import Counter
from pathlib import Path


def read(path):
    return {r[0]: r[1].strip().lower() for r in list(csv.reader(open(path)))[1:] if len(r) > 1 and r[1].strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="audit")
    a = ap.parse_args()
    d = Path(a.dir)
    det = {Path(p).stem.split("_", 1)[1]: read(p) for p in glob.glob(str(d / "labels_[!t]*.csv"))}
    ab = {Path(p).stem.split("_", 1)[1]: read(p) for p in glob.glob(str(d / "ab_[!tk]*.csv"))}
    key = {r["id"]: r for r in csv.DictReader(open(d / "ab_key.csv"))}
    lines = []
    if det:
        per = {r: sum(v == "damage" for v in l.values()) / max(len(l), 1) for r, l in det.items()}
        ids = set().union(*[set(l) for l in det.values()])
        maj = [Counter(l.get(i) for l in det.values() if l.get(i)).most_common(1)[0][0] for i in ids]
        lines.append(f"Detector precision (share labelled real damage): per rater {', '.join(f'{r} {p:.2f}' for r, p in per.items())}; "
                     f"majority vote {sum(m == 'damage' for m in maj) / len(maj):.2f} over {len(maj)} detections "
                     f"({sum(m == 'design' for m in maj)} design, {sum(m == 'unsure' for m in maj)} unsure)")
    if ab:
        wins = Counter()
        for r, l in ab.items():
            for i, v in l.items():
                wins[key[i][v.upper()] if v.upper() in "ABC" and v else "same"] += 1
        tot = sum(wins.values())
        lines.append("Blind preference (all raters, all items): " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in wins.most_common()) + f" (n={tot})")
    (d / "audit_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
