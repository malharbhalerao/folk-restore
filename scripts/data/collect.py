"""collect an open-licence dataset of Indian folk paintings from public museum / Commons APIs.

    python scripts/data/collect.py find-categories --term Madhubani      # (optional) discover Commons category names
    python scripts/data/collect.py collect --contact you@example.com     # download into data/raw/<style>/

Sources (all free, no API key):  Wikimedia Commons, The Met (public domain only), Cleveland Museum of Art (CC0 only),
Art Institute of Chicago (public domain only).  Every file is logged with its licence in data/sources.csv.
Re-running is safe: files that already exist are skipped, so you can interrupt and resume.
Edit STYLES below to add category names / search terms.
"""
import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import requests

from folk_restore import append_sources, list_images

STYLES = {
    "madhubani": {
        "commons_categories": ["Category:Madhubani paintings", "Category:Madhubani art", "Category:Mithila painting"],
        "commons_search": ["Madhubani painting", "Mithila painting"],
        "museum_queries": ["Madhubani", "Mithila"],
    },
    "warli": {
        "commons_categories": ["Category:Warli painting", "Category:Warli art", "Category:Warli paintings"],
        "commons_search": ["Warli painting"],
        "museum_queries": ["Warli"],
    },
    "pattachitra": {
        "commons_categories": ["Category:Pattachitra", "Category:Patachitra", "Category:Pattachitra paintings"],
        "commons_search": ["Pattachitra", "Patachitra painting"],
        "museum_queries": ["Pattachitra", "Patachitra"],
    },
    "kalamkari": {
        "commons_categories": ["Category:Kalamkari", "Category:Kalamkari paintings", "Category:Kalamkari art"],
        "commons_search": ["Kalamkari"],
        "museum_queries": ["Kalamkari"],
    },
}

KEYWORDS = {
    "madhubani": ["madhubani", "mithila", "maithil"],
    "warli": ["warli", "varli"],
    "pattachitra": ["pattachitra", "patachitra", "pata chitra", "pata painting"],
    "kalamkari": ["kalamkari", "kalamakari"],
}


def relevant(style, *texts):
    """Museum full-text search is fuzzy; keep only records whose metadata really mentions the style."""
    blob = " ".join(str(t) for t in texts).lower()
    return any(re.search(r"\b" + re.escape(k) + r"\b", blob) for k in KEYWORDS[style])


COMMONS = "https://commons.wikimedia.org/w/api.php"
MET = "https://collectionapi.metmuseum.org/public/collection/v1"
CMA = "https://openaccess-api.clevelandart.org/api/artworks/"
AIC = "https://api.artic.edu/api/v1"


class Collector:
    def __init__(self, out, contact, min_side, sleep):
        self.out, self.raw = Path(out), Path(out) / "raw"
        self.meta = self.out / "sources.csv"
        self.min_side, self.sleep = min_side, sleep
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"folk-restore-coursework/0.1 ({contact})"
        self.new = defaultdict(int)  # (style, source) -> newly saved
        self.streak = defaultdict(int)  # source -> consecutive failed downloads

    # ---- http
    def get_json(self, url, params=None, tries=5):
        for k in range(tries):
            try:
                r = self.s.get(url, params=params, timeout=40)
            except requests.RequestException as e:
                print(f"    network error ({e}); retry {k + 1}/{tries}")
                time.sleep(2 ** k)
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 2 ** (k + 1)))
                print(f"    HTTP {r.status_code}; waiting {wait:.0f}s")
                time.sleep(min(wait, 60))
                continue
            try:
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                print(f"    bad response: {e}")
                return None
        return None

    def download(self, url, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        for k in range(4):
            try:
                with self.s.get(url, stream=True, timeout=60) as r:
                    if r.status_code in (429, 500, 502, 503, 504):
                        time.sleep(min(float(r.headers.get("Retry-After", 2 ** (k + 1))), 60))
                        continue
                    if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("image"):
                        return False
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 16):
                            f.write(chunk)
                tmp.rename(dest)
                time.sleep(self.sleep)
                return True
            except requests.RequestException:
                time.sleep(2 ** k)
        tmp.unlink(missing_ok=True)
        return False

    # ---- bookkeeping
    def blocked(self, source, n=8):
        if self.streak[source] >= n:
            print(f"    {source}: {n} downloads in a row failed (blocked or rate-limited); giving up on this source")
            return True
        return False

    def n_existing(self, style):
        return len(list_images(self.raw / style))

    def exists(self, style, stem):
        return any((self.raw / style).glob(stem + ".*"))

    def save(self, style, source, stem, url, license_, title, credit=""):
        """Download, verify it decodes and is big enough, log provenance. Returns True if kept."""
        if self.exists(style, stem):
            return False
        ext = ".png" if url.lower().split("?")[0].endswith(".png") else ".jpg"
        dest = self.raw / style / f"{stem}{ext}"
        if not self.download(url, dest):
            self.streak[source] += 1
            return False
        self.streak[source] = 0
        img = cv2.imread(str(dest), cv2.IMREAD_REDUCED_COLOR_2)
        full = cv2.imread(str(dest), cv2.IMREAD_COLOR) if img is None else None
        shape = (img.shape[0] * 2, img.shape[1] * 2) if img is not None else (full.shape[:2] if full is not None else None)
        if shape is None or min(shape) < self.min_side:
            dest.unlink(missing_ok=True)
            return False
        append_sources(self.meta, [dict(file=f"{style}/{dest.name}", style=style, source=source, url=url,
                                        license=license_, title=title, credit=credit)])
        self.new[(style, source)] += 1
        return True


def clean(s, n=60):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:n]


# ----------------------------------------------------------------------------- Wikimedia Commons
def commons_category_files(col, cat, depth, seen):
    titles, params = [], dict(action="query", format="json", formatversion=2, list="categorymembers",
                              cmtitle=cat, cmtype="file|subcat", cmlimit=500)
    while True:
        d = col.get_json(COMMONS, params)
        if not d:
            break
        for m in d.get("query", {}).get("categorymembers", []):
            if m.get("ns") == 6:
                titles.append(m["title"])
            elif m.get("ns") == 14 and depth > 0 and m["title"] not in seen:
                seen.add(m["title"])
                titles += commons_category_files(col, m["title"], depth - 1, seen)
        if "continue" in d:
            params.update(d["continue"])
        else:
            break
    return titles


def commons_search_files(col, term, limit):
    titles, params = [], dict(action="query", format="json", formatversion=2, list="search",
                              srsearch=term, srnamespace=6, srlimit=50)
    while len(titles) < limit:
        d = col.get_json(COMMONS, params)
        if not d:
            break
        titles += [h["title"] for h in d.get("query", {}).get("search", [])]
        if "continue" in d:
            params.update(d["continue"])
        else:
            break
    return titles[:limit]


def collect_commons(col, style, cfg, limit, depth=1, search_limit=100, width=2400):
    titles, seen = [], set()
    for cat in cfg["commons_categories"]:
        got = commons_category_files(col, cat, depth, seen)
        print(f"    {cat}: {len(got)} files" if got else f"    {cat}: nothing (check name with find-categories)")
        titles += got
    for term in cfg["commons_search"]:
        titles += commons_search_files(col, term, search_limit)
    titles = list(dict.fromkeys(titles))
    print(f"    {len(titles)} candidate files")
    saved = 0
    for i in range(0, len(titles), 50):
        if saved >= limit:
            break
        d = col.get_json(COMMONS, dict(action="query", format="json", formatversion=2, prop="imageinfo",
                                       titles="|".join(titles[i:i + 50]), iiprop="url|size|mime|extmetadata",
                                       iiurlwidth=width, iiextmetadatafilter="LicenseShortName|Artist|Credit"))
        for p in (d or {}).get("query", {}).get("pages", []):
            if saved >= limit:
                break
            ii = (p.get("imageinfo") or [None])[0]
            if not ii or ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if min(ii.get("width", 0), ii.get("height", 0)) < col.min_side:
                continue
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "unknown")
            artist = re.sub(r"<[^>]+>", "", str(meta.get("Artist", {}).get("value", "")))[:120]
            stem = "commons_" + clean(re.sub(r"\.\w{2,4}$", "", re.sub(r"^File:", "", p["title"])))
            url = ii.get("thumburl") or ii.get("url")
            if url and col.save(style, "wikimedia-commons", stem, url, lic, p["title"], artist):
                saved += 1
    return saved


# ----------------------------------------------------------------------------- The Met (public domain)
def collect_met(col, style, cfg, limit, max_candidates=300):
    ids = []
    for q in cfg["museum_queries"]:
        d = col.get_json(MET + "/search", dict(q=q, hasImages="true"))
        ids += (d or {}).get("objectIDs") or []
    ids = list(dict.fromkeys(ids))[:max_candidates]
    print(f"    {len(ids)} candidate objects" + ("" if ids else "  (search returned nothing or was refused; skipping this source)"))
    saved = 0
    for oid in ids:
        if saved >= limit:
            break
        obj = col.get_json(f"{MET}/objects/{oid}")
        time.sleep(0.1)
        if not obj or not obj.get("isPublicDomain") or not obj.get("primaryImage"):
            continue
        if not relevant(style, obj.get("title"), obj.get("culture"), obj.get("objectName"), obj.get("medium"),
                        obj.get("classification"), [t.get("term") for t in (obj.get("tags") or [])]):
            continue
        if col.blocked("met-museum"):
            break
        if col.save(style, "met-museum", f"met_{oid}", obj["primaryImage"], "Public domain (Met Open Access)",
                    obj.get("title", ""), obj.get("culture") or obj.get("artistDisplayName") or ""):
            saved += 1
    return saved


# ----------------------------------------------------------------------------- Cleveland Museum of Art (CC0)
def collect_cleveland(col, style, cfg, limit):
    saved = 0
    for q in cfg["museum_queries"]:
        skip = 0
        while saved < limit:
            d = col.get_json(CMA, dict(q=q, has_image=1, cc0=1, limit=100, skip=skip))
            data = (d or {}).get("data") or []
            if not data:
                break
            for a in data:
                if saved >= limit:
                    break
                if a.get("share_license_status") != "CC0":
                    continue
                if not relevant(style, a.get("title"), a.get("culture"), a.get("description"), a.get("tombstone"),
                                a.get("technique"), a.get("type")):
                    continue
                if col.blocked("cleveland-museum"):
                    return saved
                im = a.get("images") or {}
                url = ((im.get("print") or im.get("web") or {}).get("url"))
                if url and col.save(style, "cleveland-museum", f"cma_{a.get('id')}", url, "CC0",
                                    a.get("title", ""), a.get("creditline", "")[:120]):
                    saved += 1
            skip += 100
            if skip >= (d.get("info") or {}).get("total", 0):
                break
    return saved


# ----------------------------------------------------------------------------- Art Institute of Chicago (public domain)
def collect_artic(col, style, cfg, limit):
    saved = 0
    for q in cfg["museum_queries"]:
        page = 1
        while saved < limit:
            d = col.get_json(AIC + "/artworks/search", dict(q=q, limit=100, page=page,
                             fields="id,title,image_id,is_public_domain,artist_display,place_of_origin,description,style_title,classification_title,medium_display"))
            data = (d or {}).get("data") or []
            if not data:
                break
            iiif = (d.get("config") or {}).get("iiif_url", "https://www.artic.edu/iiif/2")
            for a in data:
                if saved >= limit:
                    break
                if not a.get("is_public_domain") or not a.get("image_id"):
                    continue
                if not relevant(style, a.get("title"), a.get("artist_display"), a.get("place_of_origin"), a.get("description"),
                                a.get("style_title"), a.get("classification_title"), a.get("medium_display")):
                    continue
                if col.blocked("art-institute-chicago"):
                    return saved
                url = f"{iiif}/{a['image_id']}/full/843,/0/default.jpg"
                if col.save(style, "art-institute-chicago", f"aic_{a['id']}", url, "Public domain (AIC)",
                            a.get("title", ""), (a.get("artist_display") or "")[:120]):
                    saved += 1
            if page >= min((d.get("pagination") or {}).get("total_pages", 1), 10):  # API caps results at 1000
                break
            page += 1
    return saved


SOURCES = {"commons": collect_commons, "met": collect_met, "cleveland": collect_cleveland, "artic": collect_artic}


# ----------------------------------------------------------------------------- commands
def cmd_find_categories(a):
    col = Collector(a.out, a.contact, a.min_side, a.sleep)
    d = col.get_json(COMMONS, dict(action="query", format="json", formatversion=2, list="search",
                                   srsearch=a.term, srnamespace=14, srlimit=30))
    hits = [h["title"] for h in (d or {}).get("query", {}).get("search", [])]
    print("\n".join(hits) if hits else "no categories found")
    print("\nPaste useful names (with the 'Category:' prefix) into STYLES[...]['commons_categories'] in this file.")


def cmd_collect(a):
    col = Collector(a.out, a.contact, a.min_side, a.sleep)
    for style in a.styles:
        cfg = STYLES[style]
        print(f"\n=== {style} ===")
        for src in a.sources:
            room = a.max_per_style - col.n_existing(style)
            limit = min(a.per_source, room)
            if limit <= 0:
                print(f"  [{src}] style already has {col.n_existing(style)} images (cap {a.max_per_style}); skipping")
                continue
            print(f"  [{src}] up to {limit} images ...")
            try:
                kwargs = {"depth": a.commons_depth} if src == "commons" else {}
                n = SOURCES[src](col, style, cfg, limit, **kwargs)
                print(f"  [{src}] saved {n} new images")
            except Exception as e:  # noqa: BLE001  one failing source must not stop the others
                print(f"  [{src}] FAILED: {type(e).__name__}: {e}  (continuing)")
    print("\n=== summary (images now in data/raw) ===")
    for style in a.styles:
        by = defaultdict(int)
        for p in list_images(col.raw / style):
            by[p.name.split("_")[0]] += 1
        print(f"  {style:12s} total {sum(by.values()):4d}   {dict(by)}")
    print(f"\nprovenance + licences: {col.meta}")
    print("Next: look at the images (scripts/data/contact_sheet.py), move 5-10 damaged ones to data/real_damaged, then run step 1.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("collect", cmd_collect), ("find-categories", cmd_find_categories)):
        p = sub.add_parser(name)
        p.add_argument("--out", default="data")
        p.add_argument("--contact", default="student-project", help="your email; Wikimedia asks for a contact in the User-Agent")
        p.add_argument("--min_side", type=int, default=800, help="drop images whose short side is smaller (px)")
        p.add_argument("--sleep", type=float, default=0.5, help="pause between downloads (be polite)")
        p.set_defaults(fn=fn)
        if name == "collect":
            p.add_argument("--styles", nargs="+", default=list(STYLES), choices=list(STYLES))
            p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
            p.add_argument("--per_source", type=int, default=30, help="max new images per style per source")
            p.add_argument("--max_per_style", type=int, default=120)
            p.add_argument("--commons_depth", type=int, default=1, help="how many sub-category levels to follow")
        else:
            p.add_argument("--term", required=True)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
