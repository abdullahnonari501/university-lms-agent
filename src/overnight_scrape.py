"""Overnight full-crawl per OVERNIGHT_SCRAPE_SPEC.md.

Sitemap-seeded harvest of giki.edu.pk into data/raw/{pages,departments,
courses,personnel}/. Resumable, checkpointed, and self-stopping per the spec's
STOP conditions. Reuses scraper.py's build_session()/load_robots()/
extract_title_and_text() unchanged -- this file only adds crawl orchestration,
it does not touch TLS handling or the Phase-1 script.
"""

import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scraper import USER_AGENT, build_session, extract_title_and_text, load_robots

BASE_URL = "https://giki.edu.pk/"
REQUEST_DELAY_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 15
CONNECT_TIMEOUT_SECONDS = 10
MAX_REQUEST_SECONDS = 90  # hard wall-clock cap per request, see fetch()
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
MAX_REDIRECTS = 5
MIN_WORD_COUNT = 100
MAX_DATA_BYTES = 500 * 1024 * 1024
MAX_FAILURE_RATE = 0.15
MIN_ATTEMPTS_BEFORE_RATE_CHECK = 20
DOC_EXTENSIONS = (".pdf", ".doc", ".docx", ".xlsx")
CHECKPOINT_PER_CATEGORY = 8  # ~32 total, spans every category (see run_summary note)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = DATA_DIR / "logs"
MANIFEST_PATH = RAW_DIR / "manifest.json"
FAILED_LOG = LOG_DIR / "failed_urls.txt"
SKIPPED_LOG = LOG_DIR / "skipped_thin.txt"
SUMMARY_LOG = LOG_DIR / "run_summary.txt"
FOUND_DOCS_LOG = LOG_DIR / "found_documents.txt"
CHECKPOINT_MARKER = LOG_DIR / ".checkpoint_passed"

# Priority order per spec. post-sitemap*/tribe_events/category/portfolio*/
# *_category are excluded entirely (news/events -- separate future phase).
SITEMAPS = {
    "pages": ["page-sitemap1.xml", "page-sitemap2.xml", "page-sitemap3.xml"],
    "departments": ["department-sitemap.xml"],
    "courses": ["course-sitemap1.xml", "course-sitemap2.xml", "course-sitemap3.xml", "course-sitemap4.xml"],
    "personnel": ["personnel-sitemap1.xml", "personnel-sitemap2.xml"],
}
CATEGORY_ORDER = ["pages", "departments", "courses", "personnel"]

NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


class StopCrawl(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def fetch(
    session: requests.Session,
    url: str,
    max_bytes: int = MAX_RESPONSE_BYTES,
    max_seconds: float = MAX_REQUEST_SECONDS,
) -> requests.Response:
    """GET a URL under a hard wall-clock cap.

    max_bytes/max_seconds default to the HTML-crawl limits; Phase 2 raises them
    for PDF downloads, where a 20 MB prospectus is legitimate rather than a
    runaway response.

    requests' `timeout` only bounds the gap between bytes, never total elapsed
    time, so a server that trickles the body (or a long redirect chain) hangs
    right past it -- this crawl stalled 15+ minutes on a single URL, twice,
    with timeout=15 set. Stream the body and abort once the cap is exceeded.
    """
    deadline = time.monotonic() + max_seconds
    resp = session.get(
        url,
        timeout=(CONNECT_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS),
        stream=True,
    )
    try:
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            if time.monotonic() > deadline:
                raise requests.exceptions.Timeout(
                    f"exceeded {max_seconds}s wall-clock cap"
                )
            total += len(chunk)
            if total > max_bytes:
                raise requests.exceptions.RequestException(
                    f"response body exceeded {max_bytes} bytes"
                )
            chunks.append(chunk)
    finally:
        resp.close()

    resp._content = b"".join(chunks)
    resp._content_consumed = True
    return resp


def fetch_sitemap_urls(session: requests.Session, sitemap_name: str) -> list[str]:
    url = urljoin(BASE_URL, sitemap_name)
    print(f"  fetching sitemap {sitemap_name} ...", flush=True)
    resp = fetch(session, url)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    return [el.find(f"{NS}loc").text for el in root.findall(f"{NS}url")]


def build_url_list(session: requests.Session) -> list[tuple[str, str]]:
    all_urls: list[tuple[str, str]] = []
    seen: set[str] = set()
    for category in CATEGORY_ORDER:
        for sitemap_name in SITEMAPS[category]:
            for u in fetch_sitemap_urls(session, sitemap_name):
                if u not in seen:
                    seen.add(u)
                    all_urls.append((u, category))
            time.sleep(REQUEST_DELAY_SECONDS)
    return all_urls


def build_checkpoint_sample(url_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deliberately span every category for the mandatory checkpoint.

    Strict priority order (all 507 'pages' before any other category) would
    make the first 30 URLs 100% 'pages', never satisfying the spec's
    'across at least 3 categories' requirement. Conservative resolution:
    sample from every category up front, independent of overall priority
    order, and note this deviation in run_summary.txt.
    """
    by_cat: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for u, c in url_list:
        by_cat[c].append((u, c))
    sample = []
    for cat in CATEGORY_ORDER:
        sample.extend(by_cat[cat][:CHECKPOINT_PER_CATEGORY])
    return sample


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-").lower()
    return slug or "index"


def load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return []


def already_saved_urls(manifest: list[dict]) -> set[str]:
    done = set()
    for e in manifest:
        out_path = RAW_DIR / e["category"] / f"{e['slug']}.txt"
        if out_path.exists():
            done.add(e["url"])
    return done


def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def collect_documents(soup: BeautifulSoup, page_url: str) -> list[str]:
    found = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().split("?")[0].endswith(DOC_EXTENSIONS):
            found.append(urljoin(page_url, href))
    return found


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def process_url(
    session: requests.Session,
    robots,
    url: str,
    category: str,
    manifest: list[dict],
    stats: dict,
) -> int | None:
    """Fetch, extract, save or skip one URL. Returns the HTTP status code
    (or None if the request never got a response)."""
    if not robots.can_fetch(USER_AGENT, url):
        stats["robots_skipped"] += 1
        return None

    stats["fetch_attempts"] += 1
    status = None
    try:
        resp = fetch(session, url)
        status = resp.status_code
        resp.raise_for_status()
    except requests.RequestException as exc:
        stats["fetch_failures"] += 1
        append_line(FAILED_LOG, f"{url}\t{category}\t{exc}")
        return status
    finally:
        time.sleep(REQUEST_DELAY_SECONDS)

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        stats["non_html_skipped"] += 1
        return status

    soup = BeautifulSoup(resp.text, "html.parser")

    for doc_url in collect_documents(soup, url):
        append_line(FOUND_DOCS_LOG, f"{doc_url}\t{url}")

    title, text = extract_title_and_text(soup)
    word_count = len(text.split())

    if word_count < MIN_WORD_COUNT:
        stats["thin_skipped"] += 1
        append_line(SKIPPED_LOG, f"{url}\t{category}\t{word_count} words")
        return status

    slug = slugify(url)
    out_dir = RAW_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.txt").write_text(text, encoding="utf-8")

    manifest.append(
        {
            "url": url,
            "title": title,
            "slug": slug,
            "category": category,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "word_count": word_count,
        }
    )
    stats["saved"] += 1
    stats[f"saved_{category}"] += 1
    print(f"[{category}] saved ({stats['saved']}): {url}", flush=True)

    if len(manifest) % 25 == 0:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return status


def check_stop_conditions(status: int | None, stats: dict) -> None:
    if status == 429:
        raise StopCrawl(f"HTTP 429 rate-limited at {datetime.now(timezone.utc).isoformat()}")

    if status == 403:
        stats["consecutive_403"] += 1
        if stats["had_success"] and stats["consecutive_403"] >= 3:
            raise StopCrawl("3+ consecutive HTTP 403s after prior success -- likely blocked")
    elif status is not None and 200 <= status < 300:
        stats["consecutive_403"] = 0
        stats["had_success"] = True

    if stats["fetch_attempts"] >= MIN_ATTEMPTS_BEFORE_RATE_CHECK:
        rate = stats["fetch_failures"] / stats["fetch_attempts"]
        if rate > MAX_FAILURE_RATE:
            raise StopCrawl(f"failure rate {rate:.1%} exceeds {MAX_FAILURE_RATE:.0%} threshold")

    if stats["saved"] and stats["saved"] % 25 == 0:
        du = dir_size_bytes(DATA_DIR)
        if du > MAX_DATA_BYTES:
            raise StopCrawl(f"data/ exceeds 500MB ({du / 1e6:.1f} MB)")


def write_run_summary(stats: dict, url_list: list[tuple[str, str]], start_time: float) -> None:
    targets = defaultdict(int)
    for _, c in url_list:
        targets[c] += 1

    manifest = load_manifest()
    saved_by_cat = defaultdict(int)
    for e in manifest:
        saved_by_cat[e["category"]] += 1

    total_size = dir_size_bytes(DATA_DIR)
    largest = sorted(RAW_DIR.rglob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)[:5]
    failed_count = FAILED_LOG.read_text(encoding="utf-8").count("\n") if FAILED_LOG.exists() else 0
    thin_count = SKIPPED_LOG.read_text(encoding="utf-8").count("\n") if SKIPPED_LOG.exists() else 0
    doc_count = FOUND_DOCS_LOG.read_text(encoding="utf-8").count("\n") if FOUND_DOCS_LOG.exists() else 0
    elapsed_min = (time.monotonic() - start_time) / 60

    lines = [
        f"Run finished: {datetime.now(timezone.utc).isoformat()}",
        f"Total run time: {elapsed_min:.1f} minutes",
        "",
        "Pages saved per category (target vs actual):",
    ]
    for cat in CATEGORY_ORDER:
        lines.append(f"  {cat}: {saved_by_cat[cat]} / {targets[cat]}")
    lines += [
        "",
        f"Failed URLs (see failed_urls.txt): {failed_count}",
        "  Note: a URL logged here may have later succeeded on retry -- "
        "manifest.json is authoritative for what's actually saved.",
        f"Thin pages skipped (<{MIN_WORD_COUNT} words, see skipped_thin.txt): {thin_count}",
        f"Documents inventoried (see found_documents.txt): {doc_count}",
        "",
        f"Total corpus size on disk: {total_size / 1e6:.2f} MB",
        "",
        "Top 5 largest files:",
    ]
    for p in largest:
        lines.append(f"  {p.stat().st_size / 1024:.1f} KB  {p.relative_to(REPO_ROOT)}")

    if stats.get("stop_reason"):
        lines += ["", f"STOPPED EARLY: {stats['stop_reason']}"]

    lines += [
        "",
        "Ambiguity resolutions made (per instruction: conservative + noted here):",
        "  - Checkpoint sample: priority order alone would make the first 30 "
        f"URLs 100% 'pages'. Sampled {CHECKPOINT_PER_CATEGORY} per category "
        "instead so the mandatory quality checkpoint actually spans all 4 "
        "categories, then ran the real crawl in the spec's stated priority order.",
        "  - Failure-rate/403 STOP conditions exclude robots-disallowed and "
        "non-HTML skips (not server-health signals) from the denominator.",
        "  - found_documents.txt scans the full fetched HTML (not just the "
        "extracted main-content region), since document links in nav/sidebar "
        "areas are still valid Phase 2 candidates.",
        "",
        "Recommended next step for Phase 2: build the embeddings/retrieval "
        f"layer over data/raw/{{pages,departments,courses,personnel}}; "
        f"found_documents.txt lists {doc_count} candidate PDF/DOC/XLSX links "
        "for docling-based ingestion.",
    ]
    SUMMARY_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {SUMMARY_LOG}")


def main() -> None:
    start_time = time.monotonic()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    session = build_session()
    session.max_redirects = MAX_REDIRECTS  # bound redirect chains; each hop gets its own timeout
    robots = load_robots(session, BASE_URL)

    print("Fetching sitemaps...")
    url_list = build_url_list(session)
    print(f"Total target URLs: {len(url_list)}")

    manifest = load_manifest()
    done = already_saved_urls(manifest)

    stats = defaultdict(int)
    stats["consecutive_403"] = 0
    stats["had_success"] = bool(manifest)

    try:
        if not CHECKPOINT_MARKER.exists():
            print("=== CHECKPOINT PHASE (mandatory quality review before full run) ===")
            sample = build_checkpoint_sample(url_list)
            for url, category in sample:
                if url in done:
                    continue
                status = process_url(session, robots, url, category, manifest, stats)
                check_stop_conditions(status, stats)
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"\nCheckpoint batch done: {stats['saved']} saved, "
                f"{stats['thin_skipped']} thin-skipped, {stats['fetch_failures']} failed."
            )
            print(
                "STOPPING here for the mandatory quality review. Read the "
                "designated sample files, then create "
                f"{CHECKPOINT_MARKER} and rerun this script to continue."
            )
            return

        print("=== MAIN CRAWL ===")
        remaining = [(u, c) for u, c in url_list if u not in done]
        for url, category in remaining:
            status = process_url(session, robots, url, category, manifest, stats)
            check_stop_conditions(status, stats)

        print("=== RETRY PASS (failed URLs, once) ===")
        done_now = already_saved_urls(manifest)
        retry_targets = [(u, c) for u, c in url_list if u not in done_now]
        for url, category in retry_targets:
            status = process_url(session, robots, url, category, manifest, stats)
            check_stop_conditions(status, stats)

    except StopCrawl as e:
        stats["stop_reason"] = e.reason
        print(f"\nSTOP condition triggered: {e.reason}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_run_summary(stats, url_list, start_time)


if __name__ == "__main__":
    main()
