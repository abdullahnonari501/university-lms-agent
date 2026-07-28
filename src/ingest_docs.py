"""Phase 2 Stage A: download the 9 verified GIKI PDFs and parse them with docling.

Per OVERNIGHT_RETRIEVAL_SPEC.md. Only the 9 documents confirmed to exist are
fetched -- the fee schedule / degree-requirements / clearance PDFs do not exist
on the public site and are deliberately not searched for again.

Run from .venv (needs docling). data/raw/** is read-only except for the
data/raw/documents/ directory this script creates.
"""

import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
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
# The crawler's 20 MB / 90 s caps are tuned for HTML; the prospectuses are
# legitimately larger and slower, so raise the ceiling for PDF downloads only.
MAX_PDF_BYTES = 150 * 1024 * 1024
MAX_PDF_SECONDS = 300
# Set FORCE_NO_OCR=1 to skip the OCR attempt entirely. Used for documents
# already known to deadlock in the OCR pipeline, so a retry pass does not
# burn the full per-document timeout before reaching the fallback.
FORCE_NO_OCR = os.environ.get("FORCE_NO_OCR") == "1"

# docling's model loaders contact the HuggingFace hub even when the weights are
# already cached locally. Under multiprocessing's default fork start method that
# call deadlocked every time -- 101 threads all parked in futex_wait_queue while
# holding an open socket to the HF CDN -- and it always struck the first
# document of a run, which is why the handbook failed identically with OCR on
# and off. Spawn gives the child a clean interpreter with no inherited locks,
# and offline mode keeps the loaders on the local cache instead of the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# The actual killer: docling's native thread pools deadlock on these PDFs --
# 100 threads parked in futex_wait_queue, ~28s of CPU burned then nothing,
# reproducible with OCR on and off and under both fork and spawn. Pinning the
# math libraries to a single thread makes the same document parse in 11s.
# Must be set before torch/onnxruntime are imported, so it lives at module
# level where a spawned child re-runs it on import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
    # FES-Advisory-handbook_StudentCopy.pdf is in the inventory but now 404s;
    # replaced with another student-facing policy from the same verified list.
    ("sexual-harassment-policy", "https://giki.edu.pk/wp-content/uploads/2023/09/GIK_TOR-SEXUAL-HARRASSMENT-POLICY-approved.pdf"),
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
                resp = fetch(session, candidate,
                             max_bytes=MAX_PDF_BYTES, max_seconds=MAX_PDF_SECONDS)
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


def _parse_worker(pdf_path: str, queue: mp.Queue, use_ocr: bool = True) -> None:
    """Runs in its own process so a docling hang can be killed outright."""
    try:
        from docling.document_converter import DocumentConverter

        if use_ocr:
            converter = DocumentConverter()
        else:
            # OCR pulls in RapidOCR + HuggingFace model fetches, which deadlocked
            # on the student handbook (worker blocked in futex with a CLOSE-WAIT
            # socket to the HF CDN). These PDFs have a real text layer, so the
            # OCR pass is unnecessary cost -- disable it for the fallback attempt.
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            options = PdfPipelineOptions()
            options.do_ocr = False
            # Table structure recognition is independent of OCR and must stay on:
            # the prospectus tables are the highest-value content in these PDFs.
            options.do_table_structure = True
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )

        result = converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()

        # Hand the text back through a file, not the queue. mp.Queue.put()
        # blocks once the payload exceeds the OS pipe buffer (~64 KB) and only
        # unblocks when the parent drains it -- but the parent is inside
        # proc.join(), so the two deadlock. Every document that "hung" was
        # simply larger than the buffer; the small ones fit and succeeded.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(markdown)
            queue.put(("ok", handle.name))
    except Exception as exc:  # noqa: BLE001 - report any parse failure verbatim
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def parse_pdf(pdf_path: Path, slug: str, use_ocr: bool = True) -> str | None:
    """Parse with docling under a hard per-document timeout."""
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_parse_worker, args=(str(pdf_path), queue, use_ocr), daemon=True)
    started = time.monotonic()
    proc.start()
    proc.join(PARSE_TIMEOUT_SECONDS)

    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
        mode = "ocr" if use_ocr else "no-ocr"
        print(f"  [FAIL] {slug}: docling ({mode}) timed out after {PARSE_TIMEOUT_SECONDS}s", flush=True)
        log_failure(slug, f"docling ({mode}) timeout after {PARSE_TIMEOUT_SECONDS}s")
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

    tmp_path = Path(payload)
    try:
        payload = tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    words = len(payload.split())
    if words < MIN_DOC_WORDS:
        log_failure(slug, f"parsed text too short: {words} words (< {MIN_DOC_WORDS})")
        print(f"  [FAIL] {slug}: only {words} words extracted", flush=True)
        return None

    print(f"  [ok]   {slug} parsed: {words:,} words in {elapsed:.0f}s", flush=True)
    return payload


def main() -> int:
    # Must be spawn, not fork -- see the HF_HUB_OFFLINE note above.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    session = build_session()
    session.max_redirects = MAX_REDIRECTS

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    already = {e["url"] for e in manifest}

    succeeded, failed = [], []
    for slug, url in DOCUMENTS:
        print(f"[{slug}]", flush=True)
        existing = TEXT_DIR / f"{slug}.txt"
        if existing.exists() and len(existing.read_text(encoding="utf-8").split()) >= MIN_DOC_WORDS:
            wc = len(existing.read_text(encoding="utf-8").split())
            print(f"  [skip] {slug} already parsed ({wc:,} words)", flush=True)
            succeeded.append((slug, wc))
            continue

        pdf_path = download(session, slug, url)
        if pdf_path is None:
            failed.append(slug)
            continue

        if FORCE_NO_OCR:
            text = parse_pdf(pdf_path, slug, use_ocr=False)
        else:
            text = parse_pdf(pdf_path, slug)
            if text is None:
                print(f"  [fallback] retrying {slug} with OCR disabled", flush=True)
                text = parse_pdf(pdf_path, slug, use_ocr=False)
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
