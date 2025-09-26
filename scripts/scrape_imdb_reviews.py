import argparse
import csv
import json
import os
import random
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE = "https://www.imdb.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
BASE_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}
CHECKPOINT_FILE = "checkpoint.json"

# ---------- Language detection ----------
try:
    from langdetect import detect
    HAVE_LANGDETECT = True
except Exception:
    HAVE_LANGDETECT = False

# ---------- Errors ----------
class FetchError(Exception):
    pass

def safe_sleep(min_s: float, max_s: float):
    time.sleep(random.uniform(min_s, max_s))

def zero_pad_imdb_id(imdb_int: int) -> str:
    return f"tt{int(imdb_int):07d}"

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def is_english(text: str, mode: str) -> bool:
    if mode == "any":
        return True
    if not text:
        return False
    txt = text.strip()
    if len(txt) < 40:
        ascii_ratio = sum(c.isascii() for c in txt) / max(1, len(txt))
        if ascii_ratio < 0.85:
            return False
        lower = txt.lower()
        return any(w in lower for w in [" the ", " and ", " is ", " it ", " of ", " to ", " in ", " a "])
    if HAVE_LANGDETECT:
        try:
            return detect(txt) == "en"
        except Exception:
            pass
    ascii_ratio = sum(c.isascii() for c in txt) / max(1, len(txt))
    return ascii_ratio >= 0.9

# ---------- Robust fetch with retries & 404 handling ----------
def _fetch_html_with_retries(session: requests.Session, url: str, headers: dict,
                             timeout: int = 30, retries: int = 4, base_backoff: float = 1.5) -> Optional[str]:
    """
    Returns HTML text on success.
    Returns None for HTTP 404 (treat as 'no reviews page').
    Raises FetchError for persistent failures.
    Retries on 429 and 5xx with exponential backoff + jitter.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            code = resp.status_code
            # Hard exit for 404: page truly not found
            if code == 404:
                return None
            # Retry-worthy statuses
            if code == 429 or 500 <= code < 600:
                # backoff with jitter
                sleep_s = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue
            # Other errors -> raise
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            last_exc = e
            # network hiccup/backoff
            sleep_s = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
            continue
    raise FetchError(f"Failed to fetch after {retries} attempts: {url} | last_exc={last_exc}")

def get_soup(session: requests.Session, url: str, min_delay: float, max_delay: float,
             referer: Optional[str] = None, debug_dump_path: Optional[str] = None) -> Optional[BeautifulSoup]:
    headers = dict(BASE_HEADERS)
    if referer:
        headers["Referer"] = referer
    html = _fetch_html_with_retries(session, url, headers=headers)
    if html is None:
        # 404
        return None
    safe_sleep(min_delay, max_delay)
    if debug_dump_path:
        with open(debug_dump_path, "w", encoding="utf-8") as f:
            f.write(html)
    return BeautifulSoup(html, "lxml")

def first_sel(soup: BeautifulSoup, selectors: List[str]):
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el
    return None

def extract_review(container: BeautifulSoup) -> Dict:
    # Title (modern first)
    title_el = first_sel(container, [
        "[data-testid='review-summary']",
        ".title", "a.title",
        ".lister-item-content a.title",
        "h3 a", "h3",
    ])
    title = normalize_ws(title_el.get_text(" ", strip=True)) if title_el else ""

    # Body (modern first)
    content_el = first_sel(container, [
        "[data-testid='review-text']",
        ".ipc-html-content-inner-div",
        ".text.show-more__control",
        ".content .text",
        ".ipc-html-content",
        ".lister-item-content p",
        ".content",
    ])
    content = normalize_ws(content_el.get_text(" ", strip=True)) if content_el else ""

    # No stable review_id in modern markup -> fingerprint for dedupe/resume
    fp_src = f"{title}\n{content}".encode("utf-8")
    fingerprint = hashlib.sha1(fp_src).hexdigest()

    return {
        "fingerprint": fingerprint,
        "title": title,
        "content": content,
    }

def collect_page_reviews(soup: BeautifulSoup) -> Tuple[List[Dict], bool]:
    # Modern containers; keep legacy fallback just in case
    containers = soup.select("article.user-review-item, div[data-review-id]") if soup else []
    out = [extract_review(c) for c in containers]
    # Modern page often has a "25 more" button, not data-key. Return bool if a button exists.
    has_more_btn = soup.select_one("button.ipc-see-more__button") is not None if soup else False
    return out, has_more_btn

def ensure_csv_header(path: str):
    # Ensure directory exists
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if new:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["imdb_id", "title", "content"])

def append_rows(path: str, rows: List[Dict]):
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r["imdb_id"], r["title"], r["content"]])

# --------------------- Error logging ---------------------
def log_error(path: str, ttid: str, url: str, stage: str, message: str):
    # Ensure directory exists
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    rec = {
        "ttid": ttid,
        "url": url,
        "stage": stage,
        "message": message,
        "ts": int(time.time())
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --------------------- Checkpoint ---------------------
def load_checkpoint(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "mode": None,
        "links_csv": None,
        "current_index": 0,
        "per_title_saved": {},
        "current_title": None,
        "current_seen_fps": [],  # fingerprints for current title
    }

def save_checkpoint(state: Dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# --------------------- Core scrape ---------------------
def scrape_title_featured(
    session: requests.Session,
    ttid: str,
    limit: int,
    lang_mode: str,
    min_delay: float,
    max_delay: float,
    seen_fps: Set[str],
    debug: bool = False,
) -> Tuple[List[Dict], Set[str]]:
    collected: List[Dict] = []

    base_url = f"{BASE}/title/{ttid}/reviews/"
    soup = get_soup(session, base_url, min_delay, max_delay,
                    referer=None,
                    debug_dump_path="first_page.html" if debug else None)

    # If 404 (no page) -> skip cleanly
    if soup is None:
        return [], seen_fps

    page_rows, has_more = collect_page_reviews(soup)


    kept = dropped_non_en = dropped_empty = 0
    pbar = tqdm(total=limit, desc=ttid, leave=False)

    for r in page_rows:
        fp = r.get("fingerprint")
        if not r["content"]:
            dropped_empty += 1
            continue
        if lang_mode != "any" and not is_english(r["content"], lang_mode):
            dropped_non_en += 1
            continue
        if fp in seen_fps:
            continue

        collected.append({"imdb_id": ttid, "title": r["title"], "content": r["content"]})
        seen_fps.add(fp)
        kept += 1
        pbar.update(1)
        if len(collected) >= limit:
            break

    pbar.close()
    if debug:
        print(f"[DEBUG] Kept={kept}, DroppedEmpty={dropped_empty}, DroppedNonEN={dropped_non_en}")
    return collected, seen_fps

def read_links_csv(path: str, filter_by_movies: bool = False) -> List[str]:
    import pandas as pd

    if filter_by_movies:
        # Read ratings to get movies with actual ratings
        ratings_path = Path(path).parent / "ratings.csv"
        if not ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

        ratings_df = pd.read_csv(ratings_path)
        movies_with_ratings = set(ratings_df['movieId'].unique())
        print(f"Found {len(movies_with_ratings):,} unique movies with ratings")

        # Read links and filter by intersection
        links_df = pd.read_csv(path)
        print(f"Found {len(links_df):,} total movies in links.csv")

        # Filter links to only movies that have ratings AND valid imdbId
        filtered_links = links_df[
            (links_df['movieId'].isin(movies_with_ratings)) &
            (links_df['imdbId'].notna()) &
            (links_df['imdbId'] != 0)
        ]
        print(f"Filtered to {len(filtered_links):,} movies that have both ratings and IMDb IDs")

        # Convert to ttid list
        ttids = [zero_pad_imdb_id(int(imdb_id)) for imdb_id in filtered_links['imdbId']]

    else:
        # Original behavior - just read all links with valid imdbId
        ttids = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                imdb_id = row.get("imdbId")
                if not imdb_id:
                    continue
                try:
                    ttids.append(zero_pad_imdb_id(int(imdb_id)))
                except Exception:
                    continue

    return ttids

# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="Scrape IMDb Featured reviews (English-only by default).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ttid", help="IMDb title id (e.g., tt0096895)")
    g.add_argument("--links-csv", help="Path to MovieLens links.csv (movieId,imdbId,tmdbId)")

    ap.add_argument("--limit", type=int, default=100, help="Max reviews per title (default: 100)")
    ap.add_argument("--out", default="reviews.csv", help="Output CSV path (imdb_id,title,content)")
    ap.add_argument("--lang", choices=["en", "any"], default="en", help="Filter reviews by language (default: en)")
    ap.add_argument("--min-delay", type=float, default=1.5, help="Min delay between requests (seconds)")
    ap.add_argument("--max-delay", type=float, default=3.0, help="Max delay between requests (seconds)")
    ap.add_argument("--checkpoint", default=CHECKPOINT_FILE, help="Checkpoint file path")
    ap.add_argument("--debug", action="store_true", help="Print debug counters and page info")
    ap.add_argument("--error-log", default=None,
                    help="Path to JSONL error log (default: errors.jsonl next to --out)")
    ap.add_argument("--filter-by-movies", action="store_true",
                    help="Filter links.csv to only include movies that exist in movies.csv")


    args = ap.parse_args()

    # Default error log path next to --out
    error_log = args.error_log or os.path.join(os.path.dirname(args.out) or ".", "errors.jsonl")


    ensure_csv_header(args.out)
    state = load_checkpoint(args.checkpoint)
    session = requests.Session()

    if args.ttid:
        ttid = args.ttid
        state["mode"] = "ttid"
        state["current_title"] = ttid

        already_saved = int(state.get("per_title_saved", {}).get(ttid, 0))
        seen_fps = set(state.get("current_seen_fps", []))
        to_collect = max(0, args.limit - already_saved)
        if to_collect == 0:
            print(f"{ttid}: already saved {already_saved} (>= limit).")
            return

        try:
            new_rows, updated_seen = scrape_title_featured(
                session=session,
                ttid=ttid,
                limit=to_collect,
                lang_mode=args.lang,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
                seen_fps=seen_fps,
                debug=args.debug,
            )
        except FetchError as e:
            log_error(error_log, ttid, f"{BASE}/title/{ttid}/reviews/", "fetch", str(e))
            print(f"[ERROR] {ttid}: {e}")
            return

        # write every 10
        written = 0
        for i in range(0, len(new_rows), 10):
            chunk = new_rows[i:i+10]
            append_rows(args.out, chunk)
            written += len(chunk)
            state.setdefault("per_title_saved", {})[ttid] = already_saved + written
            state["current_seen_fps"] = list(updated_seen)
            save_checkpoint(state, args.checkpoint)

        print(f"Saved +{written} (total {already_saved + written}) for {ttid} -> {args.out}")
        return

    # links.csv mode
    ttids = read_links_csv(args.links_csv, filter_by_movies=args.filter_by_movies)
    if not ttids:
        print("No valid imdbId values in links.csv.")
        return

    state["mode"] = "links"
    state["links_csv"] = os.path.abspath(args.links_csv)
    start = int(state.get("current_index", 0))

    for idx in tqdm(range(start, len(ttids)), desc="Titles", position=0):
        ttid = ttids[idx]
        state["current_title"] = ttid

        seen_fps = set(state.get("current_seen_fps", []))
        already_saved = int(state.get("per_title_saved", {}).get(ttid, 0))
        if already_saved >= args.limit:
            state["current_index"] = idx + 1
            state["current_title"] = None
            state["current_seen_fps"] = []
            save_checkpoint(state, args.checkpoint)
            continue

        to_collect = args.limit - already_saved

        try:
            new_rows, updated_seen = scrape_title_featured(
                session=session,
                ttid=ttid,
                limit=to_collect,
                lang_mode=args.lang,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
                seen_fps=seen_fps,
                debug=args.debug,
            )
        except FetchError as e:
            # Log and skip to next title; advance checkpoint so we don't retry this one automatically
            log_error(error_log, ttid, f"{BASE}/title/{ttid}/reviews/", "fetch", str(e))
            state["current_index"] = idx + 1
            state["current_title"] = None
            state["current_seen_fps"] = []
            save_checkpoint(state, args.checkpoint)
            continue

        written = 0
        for i in range(0, len(new_rows), 10):
            chunk = new_rows[i:i+10]
            append_rows(args.out, chunk)
            written += len(chunk)
            state.setdefault("per_title_saved", {})[ttid] = already_saved + written
            state["current_index"] = idx
            state["current_title"] = ttid
            state["current_seen_fps"] = list(updated_seen)
            save_checkpoint(state, args.checkpoint)

        state["current_index"] = idx + 1
        state["current_title"] = None
        state["current_seen_fps"] = []
        save_checkpoint(state, args.checkpoint)

    print(f"Done. Output -> {args.out}. Checkpoint -> {args.checkpoint} | Errors -> {error_log}")

if __name__ == "__main__":
    main()
