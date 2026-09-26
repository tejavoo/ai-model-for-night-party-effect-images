#!/usr/bin/env python3
"""Build the Night Flash unpaired image dataset from openly-licensed sources.

The target domain ("disposable camera at night") is a *visual* aesthetic, not a
text keyword, so keyword search alone returns an incoherent mix. This script
therefore works in two stages:

  1. Candidate generation - query Openverse (the CC-licensed index over Flickr
     and Wikimedia) for a broad lo-fi / night-biased pool.
  2. Pixel-level aesthetic screening - every downloaded image is measured and
     kept or rejected against explicit look criteria, so the target set is
     stylistically coherent.

Everything is cached and resumable: API responses and finished downloads are
skipped on re-runs, so the free-tier daily quota is only ever spent on new work.

A manifest is written next to the images recording license, attribution and
source URL for every kept file.

Usage:
    python scripts/scrape_dataset.py --all
    python scripts/scrape_dataset.py --domain normal  --count 1000
    python scripts/scrape_dataset.py --domain disposable --count 1000
    python scripts/scrape_dataset.py --domain validation --count 20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000

USER_AGENT = (
    "nightflash-dataset-builder/1.0 "
    "(unpaired image-to-image research dataset; contact: local)"
)
OPENVERSE = "https://api.openverse.org/v1/images/"

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "training_photos"
CACHE_ROOT = DATA_ROOT / ".cache"

PAGE_SIZE = 20            # hard anonymous cap on api.openverse.org
MAX_PAGES = 12            # results are capped at 240/query = 12 pages
LICENSE_FILTER = "cc0,pdm,by"
LONG_SIDE = 1024          # stored image size
MIN_LONG_SIDE = 320       # reject anything too small to train on
JPEG_QUALITY = 95

# ---------------------------------------------------------------------------
# Query plans
# ---------------------------------------------------------------------------
# Broad lo-fi queries have high recall but are mostly daylight; the night-biased
# queries are small but precise. Screening does the final selection.

NIGHT_BIASED_QUERIES = [
    "toy camera night",
    "night flash photography",
    "lomography night",
    "night street flash photography",
    "night party flash photography",
    "night candid flash photography",
    "disposable camera night",
    "night film photography flash",
    "night photo lomo camera",
    "night snapshot flash",
]

LOFI_BROAD_QUERIES = [
    "toy camera",
    "disposable camera",
    "cheap camera",
    "single use camera",
    "lomography",
    "holga",
    "diana camera",
    "expired film photography",
]

DISPOSABLE_QUERIES = NIGHT_BIASED_QUERIES + LOFI_BROAD_QUERIES

# Source domain: ordinary daylight/indoor photography.
NORMAL_TRAIN_QUERIES = [
    "street photography",
    "cafe interior",
    "people walking city",
    "dog in park",
    "kitchen cooking",
    "hiking trail",
    "beach day",
    "farm animals",
    "bicycle street",
    "market stall",
]

# Held out from training on purpose: docs/DATA.md warns that randomly splitting
# near-identical frames makes validation metrics meaningless.
NORMAL_VALIDATION_QUERIES = [
    "picnic park afternoon",
    "barber shop interior",
    "train station platform",
    "bakery display counter",
    "art studio workshop",
    "ski slope winter",
    "harbour boats docked",
    "campus students walking",
]

# Words in tags/title that suggest the image is already in (or near) the
# target look. Used only to prioritise download order, never to accept an image.
NIGHT_HINT_WORDS = (
    "night", "nocturnal", "nightscape", "evening", "dusk", "sunset", "twilight",
    "afterdark", "after dark", "midnight", "nighttime", "neon", "flash",
    "nightclub", "bar", "club", "party", "candlelit", "lowlight", "low light",
    "moonlight", "moon", "streetlight", "nightfall",
)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class RateBudget:
    """Tracks the Openverse anonymous daily request allowance."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


def curl(url: str, timeout: int = 60, binary: bool = False):
    """HTTP GET via curl.

    The project venv ships without a usable CA bundle, so urllib/requests fail
    certificate verification. curl is present and verified, so all traffic goes
    through it.
    """
    proc = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", str(timeout),
         "-A", USER_AGENT, url],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:200])
    return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Candidate harvesting
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    source_id: str
    title: str
    url: str
    foreign_landing_url: str
    creator: str
    license: str
    license_version: str
    license_url: str
    width: int
    height: int
    source: str
    query: str
    tags: list = field(default_factory=list)
    priority: int = 0

    def key(self) -> str:
        return self.source_id or self.url


def api_url(query: str, page: int) -> str:
    q = urllib.parse.quote(query)
    return (f"{OPENVERSE}?q={q}&page={page}&page_size={PAGE_SIZE}"
            f"&mature=false&license={LICENSE_FILTER}")


def _cache_path(query: str, page: int) -> Path:
    slug = hashlib.sha1(f"{query}|{page}|{LICENSE_FILTER}".encode()).hexdigest()[:16]
    return CACHE_ROOT / "openverse" / f"{slug}.json"


def cached_page(query: str, page: int) -> dict | None:
    """Return a cached API page, or None if it still has to be fetched."""
    path = _cache_path(query, page)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        path.unlink(missing_ok=True)
        return None


def fetch_page(query: str, page: int) -> dict:
    """Fetch one API page and cache it."""
    data = json.loads(curl(api_url(query, page)))
    path = _cache_path(query, page)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _make_candidate(item: dict, query: str) -> Candidate | None:
    url = item.get("url")
    if not url:
        return None
    tags = [t.get("name", "") for t in (item.get("tags") or []) if isinstance(t, dict)]
    blob = f"{item.get('title') or ''} {' '.join(tags)}".lower()
    hits = sum(1 for w in NIGHT_HINT_WORDS if w in blob)
    return Candidate(
        source_id=item.get("id", ""),
        title=(item.get("title") or "").strip()[:200],
        url=url,
        foreign_landing_url=item.get("foreign_landing_url", ""),
        creator=(item.get("creator") or "unknown").strip()[:120],
        license=item.get("license", ""),
        license_version=item.get("license_version", ""),
        license_url=item.get("license_url", ""),
        width=item.get("width") or 0,
        height=item.get("height") or 0,
        source=item.get("source", ""),
        query=query,
        tags=tags[:24],
        priority=hits,
    )


def harvest(queries: list[str], budget: RateBudget, log) -> list[Candidate]:
    """Paginate every query, stopping when the daily budget is exhausted.

    Cached pages are free: quota is only spent on pages not already on disk, so
    re-runs and top-ups do not re-buy the same results.
    """
    seen: set[str] = set()
    out: list[Candidate] = []
    fetched = 0

    for query in queries:
        exhausted = False
        for page in range(1, MAX_PAGES + 1):
            data = cached_page(query, page)
            if data is None:
                if not budget.take():
                    log(f"  ! request budget exhausted during {query!r} p{page}")
                    return out
                try:
                    data = fetch_page(query, page)
                    fetched += 1
                except Exception as exc:
                    log(f"  ! page fetch failed {query!r} p{page}: {exc}")
                    time.sleep(2.0)
                    break
                # Anonymous burst allowance is 20/min.
                time.sleep(3.2)
            results = data.get("results") or []
            if not results:
                exhausted = True
                break
            for item in results:
                cand = _make_candidate(item, query)
                if cand is None:
                    continue
                k = cand.key()
                if k in seen:
                    continue
                seen.add(k)
                out.append(cand)
        log(f"  {query!r}: {len(seen)} unique so far"
            + (" (exhausted)" if exhausted else ""))

    log(f"  {fetched} new API requests, {budget.remaining} left today")
    # Night-hinted candidates first: better precision per downloaded byte.
    out.sort(key=lambda c: -c.priority)
    return out


# ---------------------------------------------------------------------------
# Visual metrics
# ---------------------------------------------------------------------------

def image_metrics(img: Image.Image) -> dict:
    """Cheap perceptual measurements of a candidate's look."""
    small = img.convert("RGB")
    small.thumbnail((320, 320), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32) / 255.0

    luma = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]

    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    sat = hsv[..., 1] / 255.0

    gx = np.abs(np.diff(luma, axis=1)).mean() if luma.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(luma, axis=0)).mean() if luma.shape[0] > 1 else 0.0
    detail = float((gx + gy) / 2.0)

    channel_means = a.reshape(-1, 3).mean(axis=0)

    return {
        "luma_mean": float(luma.mean()),
        "luma_median": float(np.median(luma)),
        "luma_std": float(luma.std()),
        "frac_dark": float((luma < 0.15).mean()),
        "frac_bright": float((luma > 0.80).mean()),
        "frac_mid": float(((luma >= 0.15) & (luma <= 0.80)).mean()),
        # A near-void background with one small bright blob is a studio macro or
        # an astronomical shot, not a flash snapshot of a scene.
        "frac_pure_black": float((luma < 0.03).mean()),
        "sat_mean": float(sat.mean()),
        "color_cast": float(channel_means.max() - channel_means.min()),
        "detail": detail,
        # Direct flash at night: a bright blown subject against a dark surround,
        # i.e. mass at both ends of the histogram. Both terms saturate so a
        # single very bright pixel cannot dominate.
        "flash_contrast": float(
            min(1.0, (luma > 0.80).mean() * 25.0)
            * min(1.0, (luma < 0.15).mean() * 3.0)
        ),
        "aspect": float(img.width / max(1, img.height)),
    }


def screen_disposable(m: dict) -> tuple[bool, float, str]:
    """Keep images that read as a cheap-film night shot with direct flash.

    NOTE: pixel statistics cannot fully certify this aesthetic. They reliably
    remove daylight frames and studio macro/astronomy shots, but keyword search
    for a rare look returns a fundamentally noisy population. See
    docs/DATASET_SOURCING.md before trusting a scraped target set.
    """
    reasons = []
    if m["frac_dark"] < 0.22:
        reasons.append("not-dark")
    if m["luma_median"] > 0.42:
        reasons.append("too-bright")
    if m["frac_bright"] < 0.002:
        reasons.append("no-flash-highlight")
    if m["sat_mean"] > 0.52:
        reasons.append("too-saturated")
    # Macro-on-black / deep-space: a void background with no mid-ground.
    if m["frac_pure_black"] > 0.70:
        reasons.append("void-background")
    if m["frac_mid"] < 0.05:
        reasons.append("no-midground")
    if m["frac_bright"] < 0.004 and m["frac_dark"] > 0.80:
        reasons.append("near-black")

    # Score: darkness + flash separation + muted colour, penalising flat frames.
    score = (
        min(1.0, m["frac_dark"] / 0.60) * 1.6
        + m["flash_contrast"] * 1.2
        + min(1.0, (0.55 - m["sat_mean"]) / 0.35) * 0.6
        + min(1.0, m["luma_std"] / 0.30) * 0.6
        + min(1.0, m["frac_mid"] / 0.30) * 0.8
        - (1.0 if "too-bright" in reasons else 0.0) * 2.0
    )
    return (not reasons), score, ",".join(reasons)


def screen_normal(m: dict) -> tuple[bool, float, str]:
    """Keep ordinary content photos that are clearly *outside* the target look.

    The domains must stay disjoint in style: an unpaired GAN learns the target
    distribution, so target-looking images sitting in source/ would teach it
    nothing. Any image that passes the target screen is refused here.
    """
    reasons = []
    if screen_disposable(m)[0]:
        reasons.append("looks-like-target")
    if m["frac_dark"] > 0.45 and m["luma_median"] < 0.22:
        reasons.append("looks-like-night-target")
    if m["luma_median"] < 0.06:
        reasons.append("near-black")
    if m["luma_median"] > 0.93:
        reasons.append("blown-out")
    if m["detail"] < 0.006:
        reasons.append("too-flat")

    score = (
        min(1.0, m["detail"] / 0.05) * 1.2
        + min(1.0, m["luma_std"] / 0.22) * 0.8
        + (1.0 if not reasons else 0.0) * 1.0
    )
    return (not reasons), score, ",".join(reasons)


# ---------------------------------------------------------------------------
# Download + validate
# ---------------------------------------------------------------------------

def load_and_normalise(raw: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return None
    if img.mode not in ("RGB", "L"):
        try:
            img = img.convert("RGB")
        except Exception:
            return None
    if min(img.size) < MIN_LONG_SIDE:
        return None
    if max(img.size) > LONG_SIDE:
        scale = LONG_SIDE / max(img.size)
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )
    return img


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def fetch_batch(cands: list[Candidate], workers: int, log) -> list[tuple[Candidate, bytes]]:
    """Download a batch of candidates concurrently."""
    results: list[tuple[Candidate, bytes]] = []
    errors = 0

    def one(c: Candidate):
        try:
            return c, curl(c.url, timeout=45, binary=True)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for out in pool.map(one, cands):
            if out is None:
                errors += 1
            else:
                results.append(out)
    if errors:
        log(f"  ({errors} downloads failed)")
    return results


def load_manifest_hashes(path: Path) -> set[int]:
    """Perceptual hashes of an already-built domain, for cross-domain dedup."""
    if not path.exists():
        return set()
    out: set[int] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("dhash"):
                try:
                    out.add(int(row["dhash"]))
                except ValueError:
                    pass
    return out


def balanced_order(candidates: list[Candidate]) -> list[Candidate]:
    """Round-robin across queries, shuffled within each.

    Taking the top-N by score selects for sharpness and contrast, which skews a
    small held-out set badly (it filled one with sharp group portraits). A
    validation set has to mirror the training distribution, so it is sampled
    rather than ranked, and no single query is allowed to dominate.
    """
    buckets: dict[str, list[Candidate]] = {}
    for c in candidates:
        buckets.setdefault(c.query, []).append(c)
    for group in buckets.values():
        group.sort(key=lambda c: hashlib.sha1(c.url.encode()).hexdigest())

    out: list[Candidate] = []
    while buckets:
        for query in list(buckets):
            group = buckets[query]
            if group:
                out.append(group.pop(0))
            if not group:
                del buckets[query]
    return out


def acquire(candidates: list[Candidate], outdir: Path, domain: str,
            want: int, budget: RateBudget, log, workers: int = 6,
            avoid_hashes: set[int] | None = None,
            select: str = "score") -> list[dict]:
    """Download, validate, screen and store images until `want` are kept.

    `avoid_hashes` lets a held-out domain refuse near-duplicates of another
    domain, so validation metrics stay honest.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.csv"
    existing_rows: list[dict] = []
    if manifest_path.exists():
        with manifest_path.open(newline="") as fh:
            existing_rows = list(csv.DictReader(fh))
    done_urls = {r["url"] for r in existing_rows}
    kept_hashes = {int(r["dhash"]) for r in existing_rows if r.get("dhash")}
    if avoid_hashes:
        kept_hashes |= avoid_hashes
    kept_sha = {r["sha256"] for r in existing_rows if r.get("sha256")}

    screen = screen_disposable if domain == "disposable" else screen_normal
    screen_name = "disposable" if domain == "disposable" else "normal"

    rows = list(existing_rows)
    stats = {"downloaded": 0, "decode_fail": 0, "too_small": 0,
             "dup": 0, "rejected": 0, "kept": len(existing_rows)}
    kept = len(existing_rows)
    seq = len(existing_rows)
    rejected_path = CACHE_ROOT / f"rejected_{domain}.csv"

    def flush():
        write_manifest(manifest_path, rows)

    pending = [c for c in candidates if c.url not in done_urls]
    if select == "spread":
        pending = balanced_order(pending)
    BATCH = 40
    for start in range(0, len(pending), BATCH):
        if kept >= want:
            break
        batch = pending[start:start + BATCH]
        fetched = fetch_batch(batch, workers, log)
        stats["downloaded"] += len(fetched)

        # Highest scoring images first so an early stop still leaves a
        # well-spread set rather than whatever the query order happened to give.
        scored = []
        for cand, raw in fetched:
            img = load_and_normalise(raw)
            if img is None:
                stats["decode_fail"] += 1
                continue
            m = image_metrics(img)
            ok, score, why = screen(m)
            if not ok:
                stats["rejected"] += 1
                append_reject(rejected_path, cand, m, why, screen_name)
                continue
            scored.append((score, cand, img, m))

        if select == "score":
            scored.sort(key=lambda t: -t[0])
        for score, cand, img, m in scored:
            if kept >= want:
                break
            sha = hashlib.sha256(
                np.asarray(img.convert("RGB")).tobytes()
            ).hexdigest()[:32]
            if sha in kept_sha:
                stats["dup"] += 1
                continue
            h = dhash(img)
            if any(hamming(h, other) <= 6 for other in kept_hashes):
                stats["dup"] += 1
                continue

            seq += 1
            name = f"{domain}_{seq:05d}.jpg"
            path = outdir / name
            img.convert("RGB").save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)

            row = {
                "filename": name,
                "domain": domain,
                "url": cand.url,
                "landing_url": cand.foreign_landing_url,
                "title": cand.title,
                "creator": cand.creator,
                "license": f"{cand.license} {cand.license_version}".strip(),
                "license_url": cand.license_url,
                "source": cand.source,
                "query": cand.query,
                "width": img.width,
                "height": img.height,
                "sha256": sha,
                "dhash": h,
                "screen": screen_name,
                "score": round(score, 4),
                **{k: round(v, 5) for k, v in m.items()},
            }
            rows.append(row)
            kept_hashes.add(h)
            kept_sha.add(sha)
            kept += 1
            stats["kept"] = kept

        if start % (BATCH * 4) == 0:
            flush()
            log(f"  {domain}: {kept}/{want} kept  {stats}")

    flush()
    log(f"  {domain}: final {kept}/{want}  {stats}")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_reject(path: Path, cand: Candidate, m: dict, why: str, screen_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"screen": screen_name, "reason": why, "url": cand.url,
           "query": cand.query, "title": cand.title,
           **{k: round(v, 5) for k, v in m.items()}}
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def contact_sheet(rows: list[dict], indir: Path, out: Path, cols: int = 10) -> None:
    """Tile kept images so the look can be eyeballed, not assumed."""
    picks = [r for r in rows if (indir / r["filename"]).exists()]
    if not picks:
        return
    picks = sorted(picks, key=lambda r: -float(r.get("score", 0)))[:cols * 6]
    rows_n = (len(picks) + cols - 1) // cols
    tw, th = 180, 180
    sheet = Image.new("RGB", (cols * tw, rows_n * th), (18, 18, 18))
    for i, r in enumerate(picks):
        try:
            im = Image.open(indir / r["filename"]).convert("RGB")
        except Exception:
            continue
        im.thumbnail((tw - 6, th - 6), Image.LANCZOS)
        x = (i % cols) * tw + (tw - im.width) // 2
        y = (i // cols) * th + (th - im.height) // 2
        sheet.paste(im, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)


def describe(rows: list[dict], domain: str) -> str:
    if not rows:
        return f"{domain}: empty"
    def avg(k):
        return sum(float(r[k]) for r in rows) / len(rows)
    lic: dict[str, int] = {}
    for r in rows:
        lic[r["license"]] = lic.get(r["license"], 0) + 1
    return (f"{domain}: n={len(rows)}  "
            f"luma_med={avg('luma_median'):.3f}  dark={avg('frac_dark'):.3f}  "
            f"bright={avg('frac_bright'):.4f}  sat={avg('sat_mean'):.3f}  "
            f"detail={avg('detail'):.4f}\n"
            f"    licenses: {lic}\n"
            f"    queries: {sorted({r['query'] for r in rows})}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DOMAIN_DIRS = {
    "normal": "normal_photos",
    "disposable": "disposable_photos",
    "validation": "validation_photos",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=[*DOMAIN_DIRS, "all"], default="all")
    ap.add_argument("--count", type=int, default=0,
                    help="images to keep per domain (0 = default)")
    ap.add_argument("--out-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--max-requests", type=int, default=160,
                    help="Openverse request budget for this run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    random.seed(args.seed)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    defaults = {"normal": 1000, "disposable": 1000, "validation": 20}
    budget = RateBudget(args.max_requests)

    def log(msg: str) -> None:
        print(msg, flush=True)

    plan = (["normal", "disposable", "validation"] if args.domain == "all"
            else [args.domain])

    for domain in plan:
        want = args.count or defaults[domain]
        outdir = args.out_root / DOMAIN_DIRS[domain]
        queries = {
            "normal": NORMAL_TRAIN_QUERIES,
            "disposable": DISPOSABLE_QUERIES,
            "validation": NORMAL_VALIDATION_QUERIES,
        }[domain]

        log(f"\n=== {domain}: target {want} ===")
        cands = harvest(queries, budget, log)
        log(f"  {len(cands)} unique candidates "
            f"({budget.remaining} requests left)")

        # The held-out set must not reuse frames that appear in training, and
        # must mirror the training distribution rather than being ranked by
        # score - otherwise a small validation set skews to sharp group photos.
        avoid = set()
        select = "score"
        if domain == "validation":
            select = "spread"
            avoid = load_manifest_hashes(
                args.out_root / DOMAIN_DIRS["normal"] / "manifest.csv")
            log(f"  refusing near-duplicates of {len(avoid)} training images")
            log(f"  sampling across {len({c.query for c in cands})} queries")

        rows = acquire(cands, outdir, domain, want, budget, log,
                       workers=args.workers, avoid_hashes=avoid, select=select)
        log(describe(rows, domain))
        contact_sheet(rows, outdir, args.out_root / f".cache" / f"sheet_{domain}.jpg")

    log(f"\nDone. Request budget used: {budget.used}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
