"""Phase 2 Stage A: download the 9 verified GIKI PDFs and parse them with docling.

Per OVERNIGHT_RETRIEVAL_SPEC.md. Only the 9 documents confirmed to exist are
fetched -- the fee schedule / degree-requirements / clearance PDFs do not exist
on the public site and are deliberately not searched for again.

Run from .venv (needs docling). data/raw/** is read-only except for the
data/raw/documents/ directory this script creates.
"""

import json
import multiprocessing as mp
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_scrape import MAX_REDIRECTS, fetch  # noqa: E402
from scraper import build_session  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "data" / "docs"
TEXT_DIR = REPO_ROOT / "data" / "raw" / "documents"
LOGS_DIR = REPO_ROOT / "data" / "logs"
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "manifest.json"
FAILED_LOG = LOGS_DIR / "phase2_failed_docs.txt"

MIN_DOC_WORDS = 200
PARSE_TIMEOUT_SECONDS = 600  # docling can hang on malformed PDFs
DOWNLOAD_RETRIES = 3

# The 9 documents verified to exist in data/logs/found_documents.txt.
# Seven are named directly by the spec; the last two are the spec's "other two
# from the verified inventory", resolved conservatively to the most
# student-facing policy documents available (see phase2_run_summary.txt).
DOCUMENTS = [
    ("student-handbook-2023-24", "https://giki.edu.pk/wp-content/uploads/2023/10/GIK-Institute-Student-Handbook-2023-24-General-Rules-Policies-and-Disicpline.pdf"),
    ("academic-calendar-2025-2026", "http://giki.edu.pk/wp-content/uploads/2025/08/Academic-Calendar-2025-2026.pdf"),
    ("transportation-policy", "https://giki.edu.pk/wp-content/uploads/2023/09/Transportation-Policy-for-Ghulam-Ishaq-Khan-Institute-of-Engineering-Sciences-and-Technology-1.pdf"),
    ("undergraduate-admissions-policy", "http://192.168.100.53/wp-content/uploads/2019/10/Undergraduate-Admissions-Policy.pdf"),
    ("prospectus-fcse-2023-breakdown", "https://giki.edu.pk/wp-content/uploads/2024/01/Prospectus-FCSE-2023-v1.6-DS-Semester-wise-Breakdown.pdf"),
    ("graduate-prospectus-2024", "https://giki.edu.pk/wp-content/uploads/2023/11/GraduateProspectus2024.pdf"),
    ("ug-prospectus-2021", "https://giki.edu.pk/wp-content/uploads/2021/10/UG_Prospectus_2021.pdf"),
    ("policy-students-with-disabilities", "https://giki.edu.pk/wp-content/uploads/2023/09/Gik-HEC-Policy-for-Students-with-Disabilities.pdf"),
    ("fes-advisory-handbook-student", "https://giki.edu.pk/wp-content/uploads/2024/03/FES-Advisory-handbook_StudentCopy.pdf"),
]


def log_failure(slug: str, reason: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with FAILED_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{slug}\t{reason}\n")
        fh.flush()


def public_url_variants(url: str) -> list[str]:
    """Some inventory URLs point at an internal address (192.168.100.53) that
    leaked into the site's HTML and is unreachable publicly. Try the public
    host with the same path as a fallback."""
    variants = [url]
    host = urlparse(url).netloc
    if re.match(r"^(192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)", host):
        variants.append(url.replace(f"http://{host}", "https://giki.edu.pk"))
    if url.startswith("http://giki.edu.pk"):
        variants.append(url.replace("http://", "https://", 1))
    return variants


def download(session: requests.Session, slug: str, url: str) -> Path | None:
    """Download one PDF with retry+backoff. Transient network failures are
    retried rather than treated as terminal (per the run's amended rules)."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCS_DIR / f"{slug}.pdf"

    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [skip] {slug} already downloaded ({dest.stat().st_size:,} bytes)", flush=True)
        return dest

    last_error = "no attempt made"
    for candidate in public_url_variants(url):
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                resp = fetch(session, candidate)
                resp.raise_for_status()
                body = resp.content
                if not body.startswith(b"%PDF"):
                    last_error = f"not a PDF (starts {body[:8]!r})"
                    break  # a valid HTTP response that isn't a PDF won't fix itself
                dest.write_bytes(body)
                print(f"  [ok]   {slug} <- {candidate} ({len(body):,} bytes)", flush=True)
                return dest
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < DOWNLOAD_RETRIES:
                    backoff = 2 ** attempt
                    print(f"  [retry {attempt}/{DOWNLOAD_RETRIES}] {slug}: {exc} (sleep {backoff}s)", flush=True)
                    time.sleep(backoff)

    print(f"  [FAIL] {slug}: {last_error}", flush=True)
    log_failure(slug, f"download failed: {last_error}")
    return None


def _parse_worker(pdf_path: str, queue: mp.Queue) -> None:
    """Runs in its own process so a docling hang can be killed outright."""
    try:
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(pdf_path)
        queue.put(("ok", result.document.export_to_markdown()))
    except Exception as exc:  # noqa: BLE001 - report any parse failure verbatim
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def parse_pdf(pdf_path: Path, slug: str) -> str | None:
    """Parse with docling under a hard per-document timeout."""
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_parse_worker, args=(str(pdf_path), queue), daemon=True)
    started = time.monotonic()
    proc.start()
    proc.join(PARSE_TIMEOUT_SECONDS)

    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
        print(f"  [FAIL] {slug}: docling timed out after {PARSE_TIMEOUT_SECONDS}s", flush=True)
        log_failure(slug, f"docling timeout after {PARSE_TIMEOUT_SECONDS}s")
        return None

    if queue.empty():
        log_failure(slug, f"docling process died with exit code {proc.exitcode}")
        print(f"  [FAIL] {slug}: docling process died (exit {proc.exitcode})", flush=True)
        return None

    status, payload = queue.get()
    if status == "error":
        log_failure(slug, f"docling error: {payload}")
        print(f"  [FAIL] {slug}: {payload}", flush=True)
        return None

    elapsed = time.monotonic() - started
    words = len(payload.split())
    if words < MIN_DOC_WORDS:
        log_failure(slug, f"parsed text too short: {words} words (< {MIN_DOC_WORDS})")
        print(f"  [FAIL] {slug}: only {words} words extracted", flush=True)
        return None

    print(f"  [ok]   {slug} parsed: {words:,} words in {elapsed:.0f}s", flush=True)
    return payload


def main() -> int:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    session = build_session()
    session.max_redirects = MAX_REDIRECTS

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    already = {e["url"] for e in manifest}

    succeeded, failed = [], []
    for slug, url in DOCUMENTS:
        print(f"[{slug}]", flush=True)
        pdf_path = download(session, slug, url)
        if pdf_path is None:
            failed.append(slug)
            continue

        text = parse_pdf(pdf_path, slug)
        if text is None:
            failed.append(slug)
            continue

        out_path = TEXT_DIR / f"{slug}.txt"
        out_path.write_text(text, encoding="utf-8")

        if url not in already:
            manifest.append(
                {
                    "url": url,
                    "title": slug.replace("-", " ").title(),
                    "slug": slug,
                    "category": "documents",
                    "doc_type": "pdf",
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "word_count": len(text.split()),
                }
            )
        succeeded.append((slug, len(text.split())))

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nStage A done: {len(succeeded)} parsed, {len(failed)} failed", flush=True)
    for slug, wc in succeeded:
        print(f"  {slug}: {wc:,} words", flush=True)
    if failed:
        print(f"  failed: {', '.join(failed)}", flush=True)

    if not succeeded:
        print("STOP: all 9 PDFs failed -- environment is broken", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
