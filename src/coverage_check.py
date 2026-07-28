"""Coverage report over the scraped corpus.

Answers two questions after a full re-scrape:
  (a) which sitemap URLs never made it into the corpus, and why
  (b) which pages actually contain Markdown tables, i.e. where the structured
      content landed

Writes data/logs/coverage_report.txt and prints a summary.
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_scrape import MAX_REDIRECTS, build_url_list  # noqa: E402
from scraper import build_session  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
LOGS_DIR = REPO_ROOT / "data" / "logs"
MANIFEST_PATH = RAW_DIR / "manifest.json"
REPORT = LOGS_DIR / "coverage_report.txt"


def table_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("|"))


def load_log_urls(name: str) -> dict[str, str]:
    """url -> reason, from a tab-separated log, last entry wins."""
    path = LOGS_DIR / name
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out[parts[0]] = parts[-1].strip()
    return out


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    in_corpus = {e["url"] for e in manifest}

    session = build_session()
    session.max_redirects = MAX_REDIRECTS
    print("Fetching sitemaps for the authoritative URL list...", flush=True)
    targets = build_url_list(session)

    thin = load_log_urls("skipped_thin.txt")
    failed = load_log_urls("failed_urls.txt")

    missing = [(u, c) for u, c in targets if u not in in_corpus]
    by_reason: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for url, cat in missing:
        if url in thin:
            by_reason[f"thin page (<100 words): {thin[url]}"].append((url, cat))
        elif url in failed:
            by_reason["fetch failed"].append((url, cat))
        else:
            by_reason["UNEXPLAINED -- not thin, not failed"].append((url, cat))

    # (b) where structured content landed
    tabled = []
    for entry in manifest:
        path = RAW_DIR / entry["category"] / f"{entry['slug']}.txt"
        if not path.exists():
            continue
        n = table_line_count(path.read_text(encoding="utf-8"))
        if n:
            tabled.append((n, entry["category"], entry["title"][:58], entry["url"]))
    tabled.sort(reverse=True)

    L: list[str] = []
    a = L.append
    a("CORPUS COVERAGE REPORT")
    a(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    a("=" * 78)
    a("")
    a(f"Sitemap target URLs ....... {len(targets):,}")
    a(f"In corpus ................. {len(in_corpus):,}  ({len(manifest):,} manifest entries)")
    a(f"Not in corpus ............. {len(missing):,}")
    a("")
    a("Corpus by category:")
    for cat, n in sorted(Counter(e["category"] for e in manifest).items()):
        a(f"  {cat:<12} {n:>6,}")
    a("")
    a("=" * 78)
    a("(a) SITEMAP URLS NOT IN THE CORPUS")
    a("=" * 78)
    if not missing:
        a("  None -- every sitemap URL is present.")
    for reason in sorted(by_reason, key=lambda r: -len(by_reason[r])):
        items = by_reason[reason]
        a("")
        a(f"  {reason}  [{len(items)} URLs]")
        for url, cat in items[:40]:
            a(f"    {cat:<11} {url}")
        if len(items) > 40:
            a(f"    ... and {len(items) - 40} more")
    a("")
    a("=" * 78)
    a(f"(b) PAGES CONTAINING MARKDOWN TABLES  [{len(tabled)} pages]")
    a("=" * 78)
    a(f"  {'rows':>5}  {'category':<12} {'title':<58} url")
    for n, cat, title, url in tabled:
        a(f"  {n:>5}  {cat:<12} {title:<58} {url}")
    a("")
    a(f"  Total table rows across the corpus: {sum(t[0] for t in tabled):,}")
    a("  Tables by category:")
    for cat, n in sorted(Counter(t[1] for t in tabled).items()):
        a(f"    {cat:<12} {n:>5} pages")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\nWrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
