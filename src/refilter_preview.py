"""Dry run of the revised page filter. Fetches only; writes no corpus, no index.

Answers, before anything is committed to:
  - how many currently-missing pages the new rules re-admit
  - a sample to eyeball for real-vs-junk
  - confirmation the WordPress theme-demo pages stay excluded

New rules (defined in overnight_scrape so the real crawl cannot diverge):
  1. MIN_WORD_COUNT floor drops to 10 -- word count is a poor proxy for value.
  2. Theme-demo pages excluded by URL identity, however long they are.
  3. Pages whose text is ~entirely site-wide boilerplate are dropped.
"""

import sys
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_scrape import (  # noqa: E402
    MAX_REDIRECTS, MIN_WORD_COUNT, REQUEST_DELAY_SECONDS, build_url_list, fetch,
    is_boilerplate, is_theme_demo,
)
from scraper import build_session, extract_title_and_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
LOGS_DIR = REPO_ROOT / "data" / "logs"
OUT = LOGS_DIR / "refilter_preview.txt"

import json  # noqa: E402
import time  # noqa: E402


def common_lines(min_share: float = 0.3) -> set[str]:
    """Lines appearing on at least this share of existing pages = boilerplate."""
    counts: Counter[str] = Counter()
    pages = 0
    for path in RAW_DIR.rglob("*.txt"):
        pages += 1
        seen = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()}
        counts.update(seen)
    if not pages:
        return set()
    return {line for line, n in counts.items() if n / pages >= min_share}


def main() -> int:
    manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    in_corpus = {e["url"] for e in manifest}

    session = build_session()
    session.max_redirects = MAX_REDIRECTS
    print("Fetching sitemaps...", flush=True)
    targets = build_url_list(session)
    missing = [(u, c) for u, c in targets if u not in in_corpus]
    print(f"{len(missing)} URLs currently missing from the corpus", flush=True)

    boiler = common_lines()
    print(f"{len(boiler)} boilerplate lines learned from the existing corpus\n", flush=True)

    readmit, still_out = [], []
    for n, (url, cat) in enumerate(missing, 1):
        if is_theme_demo(url):
            still_out.append((url, cat, 0, "theme-demo page (URL identity)"))
            continue
        try:
            resp = fetch(session, url)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                still_out.append((url, cat, 0, "not HTML"))
                continue
            title, text = extract_title_and_text(BeautifulSoup(resp.text, "html.parser"))
        except requests.RequestException as exc:
            still_out.append((url, cat, 0, f"fetch failed: {type(exc).__name__}"))
            continue
        finally:
            time.sleep(REQUEST_DELAY_SECONDS)

        words = len(text.split())
        if words < MIN_WORD_COUNT:
            still_out.append((url, cat, words, f"under {MIN_WORD_COUNT}-word floor"))
        elif is_boilerplate(text, boiler):
            still_out.append((url, cat, words, "boilerplate only, no unique content"))
        else:
            readmit.append((url, cat, words, title[:60]))

        if n % 25 == 0:
            print(f"  {n}/{len(missing)} checked, {len(readmit)} re-admitted", flush=True)

    L: list[str] = []
    a = L.append
    a("REFILTER PREVIEW - no corpus or index was modified")
    a("=" * 78)
    a(f"Currently missing .......... {len(missing)}")
    a(f"Would be RE-ADMITTED ....... {len(readmit)}")
    a(f"Would stay excluded ........ {len(still_out)}")
    a("")
    a("Re-admitted by category:")
    for cat, n in sorted(Counter(r[1] for r in readmit).items()):
        a(f"  {cat:<12} {n:>5}")
    a("")
    a("Still excluded, by reason:")
    for reason, n in sorted(Counter(s[3] for s in still_out).items(), key=lambda x: -x[1]):
        a(f"  {reason:<45} {n:>5}")
    a("")
    a(f"Theme-demo pages still excluded: "
      f"{sum(1 for s in still_out if 'theme-demo' in s[3])}")
    a(f"Any theme-demo page wrongly re-admitted: "
      f"{sum(1 for r in readmit if is_theme_demo(r[0]))}")
    a("")
    a("=" * 78)
    a("SAMPLE OF RE-ADMITTED PAGES (50, spread across the list)")
    a("=" * 78)
    step = max(1, len(readmit) // 50)
    for url, cat, words, title in readmit[::step][:50]:
        a(f"  {words:>4}w [{cat:<10}] {title}")
        a(f"        {url}")
    a("")
    a("=" * 78)
    a("ALL RE-ADMITTED (full list)")
    a("=" * 78)
    for url, cat, words, title in sorted(readmit, key=lambda r: (r[1], -r[2])):
        a(f"  {words:>4}w [{cat:<10}] {url}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
