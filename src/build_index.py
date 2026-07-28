"""Phase 2 Stage B: chunk the corpus, embed with nomic-embed-text, index in chromadb.

Per OVERNIGHT_RETRIEVAL_SPEC.md:
  - 512-token target chunks, 64-token overlap, split on paragraph/heading bounds
  - deterministic IDs "{category}/{slug}#{i}" so a rerun upserts instead of
    duplicating -- this is the resumability mechanism
  - chromadb PersistentClient at data/chroma, collection "giki", cosine space

Amended STOP rules for this run: percentage-based thresholds only fire when the
failure rate stays above threshold across a sustained window (200+ consecutive
attempts), never on a transient spike. Ambiguous/transient problems are retried
with backoff, logged, and execution continues; every near-STOP is recorded.

Usage:
    python3 src/build_index.py --selftest   # 10-chunk idempotency check
    python3 src/build_index.py              # full index
"""

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"
CHROMA_DIR = REPO_ROOT / "data" / "chroma"
LOGS_DIR = REPO_ROOT / "data" / "logs"
PROGRESS_LOG = LOGS_DIR / "phase2_progress.txt"
NEAR_STOP_LOG = LOGS_DIR / "phase2_near_stops.txt"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

TARGET_TOKENS = 512
OVERLAP_TOKENS = 64
# Tables stay whole up to this size; beyond it they split on row boundaries with
# the header repeated. The ceiling is set by what Ollama actually serves, not by
# the model card: nomic-embed-text runs here with n_ctx_slot = 2048, and inputs
# above that are rejected outright with HTTP 500 ("input is too large to
# process"). 1800 leaves headroom for the token-count approximation.
TABLE_MAX_TOKENS = 1800
EMBED_CONTEXT_LIMIT = 2048
BATCH_SIZE = 32
COLLECTION = "giki"

EMBED_RETRIES = 3
UNREACHABLE_LIMIT = 5           # consecutive unreachable -> try one server restart
SUSTAINED_WINDOW = 200          # amended: failure rate judged over this many attempts
MAX_FAILURE_RATE = 0.05
MIN_FREE_GB = 2
CHECKPOINT_CHUNKS = 200


class StopIndexing(Exception):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def record_near_stop(what: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with NEAR_STOP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{what}\n")
        fh.flush()
    log(f"  [near-stop] {what}")


def write_progress(done: int, total: int, started: float) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    elapsed = time.monotonic() - started
    with PROGRESS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(
            f"{datetime.now(timezone.utc).isoformat()}\t{done}/{total}\t"
            f"elapsed={elapsed:.0f}s\n"
        )
        fh.flush()


def free_gb() -> float:
    import shutil

    return shutil.disk_usage(REPO_ROOT).free / 1e9


# ---------------------------------------------------------------- chunking

def count_tokens(text: str) -> int:
    """Approximate token count. nomic-embed-text's exact tokenizer isn't exposed
    through the Ollama API, so use the standard ~1.33 tokens/word heuristic --
    deliberately conservative (over-estimates), so chunks land under the target
    rather than over it."""
    return int(len(text.split()) * 1.33) + 1


def is_table(block: str) -> bool:
    """True for a docling Markdown table.

    docling exports tables as pipe-delimited Markdown; a block counts as a table
    when most of its lines are pipe rows. Tables are atomic downstream -- they
    are the highest-value content in the prospectuses and a table split across
    chunks loses the row/column association that makes it answerable at all.
    """
    lines = [line for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    pipe_rows = sum(1 for line in lines if line.lstrip().startswith("|"))
    return pipe_rows >= max(2, len(lines) * 0.6)


def split_blocks(text: str) -> list[str]:
    """Split on blank lines, keeping markdown headings attached to what follows."""
    blocks: list[str] = []
    for raw in re.split(r"\n\s*\n", text):
        block = raw.strip()
        if block:
            blocks.append(block)
    return blocks


def split_oversized(block: str) -> list[str]:
    """A single paragraph longer than the target: split on sentence bounds."""
    sentences = re.split(r"(?<=[.!?])\s+", block)
    out, current = [], []
    for sentence in sentences:
        current.append(sentence)
        if count_tokens(" ".join(current)) >= TARGET_TOKENS:
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return [c for c in out if c.strip()]


def hard_split(piece: str) -> list[str]:
    """Last-resort word-boundary split with overlap.

    Sentence splitting can't help text with no sentence punctuation -- PDF table
    rows and personnel publication lists are single 2,000-token "sentences".
    Without this, chunks silently blow past the 512-token target.
    """
    words = piece.split()
    per_chunk = int(TARGET_TOKENS / 1.33)      # tokens -> words
    stride = max(1, per_chunk - int(OVERLAP_TOKENS / 1.33))
    out = []
    for start in range(0, len(words), stride):
        window = words[start : start + per_chunk]
        if window:
            out.append(" ".join(window))
        if start + per_chunk >= len(words):
            break
    return out


def pack_pieces(pieces: list[str]) -> list[str]:
    """Greedily recombine small sub-tables so we don't emit hundreds of tiny
    chunks, without ever exceeding the table budget."""
    out: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = current + [piece]
        if current and count_tokens("\n\n".join(candidate)) > TABLE_MAX_TOKENS:
            out.append("\n\n".join(current))
            current = [piece]
        else:
            current = candidate
    if current:
        out.append("\n\n".join(current))
    return out


def split_table_rows(block: str) -> list[str]:
    """Split an oversized Markdown table on row boundaries, repeating the header.

    Keeping a table whole beats splitting it -- but only while it still fits
    what the embedder accepts. The course catalogue arrived as one 142k-token
    table; Ollama rejects anything over its 2048-token slot outright, so that
    table would simply never be indexed. Splitting between rows and repeating
    the header keeps every row's column meaning intact, which is the property
    that actually matters.
    """
    # A run of tables with no blank line between them arrives as one block --
    # the course catalogue is dozens of per-course CLO grids concatenated. Split
    # at each separator row first, so every sub-table keeps its own header
    # instead of inheriting the first one in the page.
    lines = block.splitlines()
    seps = [
        i for i, line in enumerate(lines)
        if line.lstrip().startswith("|") and set(line.replace("|", "").strip()) <= {"-", " "}
        and line.strip()
    ]
    if len(seps) > 1:
        starts = [max(0, s - 1) for s in seps]
        if starts[0] > 0:
            starts.insert(0, 0)
        pieces: list[str] = []
        for n, start in enumerate(starts):
            end = starts[n + 1] if n + 1 < len(starts) else len(lines)
            sub = "\n".join(lines[start:end]).strip()
            if sub:
                pieces.extend(split_table_rows(sub))
        return pack_pieces(pieces)

    header: list[str] = []
    body_start = 0
    if lines and lines[0].lstrip().startswith("|"):
        header = [lines[0]]
        body_start = 1
        if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= {"-", " "}:
            header.append(lines[1])
            body_start = 2

    header_tokens = count_tokens("\n".join(header))
    parts: list[str] = []
    current: list[str] = []

    for line in lines[body_start:]:
        candidate = current + [line]
        if header_tokens + count_tokens("\n".join(candidate)) > TABLE_MAX_TOKENS and current:
            parts.append("\n".join(header + current))
            current = [line]
        else:
            current = candidate

    if current:
        parts.append("\n".join(header + current))
    return parts or [block]


def enforce_cap(chunks: list[str]) -> list[str]:
    """Cap every chunk at the target -- except tables, which stay whole.

    An oversized table is deliberately allowed through: nomic-embed-text has an
    8192-token context, so a long fee or course table still embeds intact, and
    keeping it whole is worth more than hitting the 512 target exactly.
    """
    final: list[str] = []
    for chunk in chunks:
        if count_tokens(chunk) > EMBED_CONTEXT_LIMIT and not is_table(chunk):
            final.extend(hard_split(chunk))
            continue
        if is_table(chunk):
            # Whole if it fits the embedder; otherwise split between rows,
            # never mid-row, with the header carried into each part.
            if count_tokens(chunk) <= TABLE_MAX_TOKENS:
                final.append(chunk)
            else:
                final.extend(split_table_rows(chunk))
        elif count_tokens(chunk) <= TARGET_TOKENS:
            final.append(chunk)
        else:
            final.extend(hard_split(chunk))

    # Nothing may exceed what the server accepts -- a rejected chunk is a chunk
    # that simply is not in the index, which is worse than an imperfect split.
    guarded: list[str] = []
    for chunk in final:
        if count_tokens(chunk) > EMBED_CONTEXT_LIMIT:
            guarded.extend(hard_split(chunk))
        else:
            guarded.append(chunk)
    return guarded


def chunk_text(text: str) -> list[str]:
    """Greedily pack blocks up to TARGET_TOKENS, carrying OVERLAP_TOKENS across."""
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append("\n\n".join(current))

    for block in split_blocks(text):
        if is_table(block):
            # A table is never merged with surrounding prose and never split:
            # it becomes exactly one chunk, whatever its size.
            flush()
            current = []
            chunks.append(block)
            continue

        if count_tokens(block) > TARGET_TOKENS:
            flush()
            current = []
            chunks.extend(split_oversized(block))
            continue

        candidate = current + [block]
        if count_tokens("\n\n".join(candidate)) > TARGET_TOKENS and current:
            flush()
            # carry a tail of the previous chunk forward as overlap
            tail, tokens = [], 0
            for prev in reversed(current):
                tokens += count_tokens(prev)
                tail.insert(0, prev)
                if tokens >= OVERLAP_TOKENS:
                    break
            current = tail + [block]
        else:
            current = candidate

    flush()
    return [c for c in enforce_cap(chunks) if c.strip()]


def build_chunk_records() -> list[dict]:
    """Every chunk of every manifest entry, with deterministic IDs."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records: list[dict] = []
    missing = 0

    for entry in manifest:
        category = entry["category"]
        slug = entry["slug"]
        path = RAW_DIR / category / f"{slug}.txt"
        if not path.exists():
            missing += 1
            continue

        text = path.read_text(encoding="utf-8")
        pieces = chunk_text(text)
        for i, piece in enumerate(pieces):
            records.append(
                {
                    "id": f"{category}/{slug}#{i}",
                    "text": piece,
                    "metadata": {
                        "source_url": entry["url"],
                        "title": entry.get("title", ""),
                        "category": category,
                        "doc_type": entry.get("doc_type", "html"),
                        "chunk_index": i,
                        "n_chunks": len(pieces),
                        "source_file": str(path.relative_to(REPO_ROOT)),
                        "scraped_at": entry.get("scraped_at", ""),
                    },
                }
            )

    if missing:
        record_near_stop(f"{missing} manifest entries had no file on disk (skipped)")
    return records


# --------------------------------------------------------------- embedding

def embed_one(text: str, timeout: int = 120) -> list[float]:
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["embedding"]


def restart_ollama() -> bool:
    """User-level restart only -- systemctl under sudo is blocked on this box."""
    record_near_stop("Ollama unreachable repeatedly; attempting one server restart")
    try:
        subprocess.Popen(
            ["setsid", "nohup", str(Path.home() / ".local/bin/ollama"), "serve"],
            stdout=open("/tmp/ollama_serve.log", "a"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  restart failed to launch: {exc}")
        return False

    for _ in range(30):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5).read()
            log("  Ollama is back up")
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


class EmbedState:
    """Tracks failures so the amended sustained-window STOP rule can be applied."""

    def __init__(self) -> None:
        self.attempts = 0
        self.failures = 0
        self.consecutive_unreachable = 0
        self.restarted = False
        self.recent: list[int] = []  # 1 = failure, 0 = success

    def record(self, ok: bool) -> None:
        self.attempts += 1
        self.recent.append(0 if ok else 1)
        if len(self.recent) > SUSTAINED_WINDOW:
            self.recent.pop(0)
        if not ok:
            self.failures += 1

    def check_sustained_failure(self) -> None:
        """Amended rule: only STOP when the rate stays high across a full window."""
        if len(self.recent) < SUSTAINED_WINDOW:
            return
        rate = sum(self.recent) / len(self.recent)
        if rate > MAX_FAILURE_RATE:
            raise StopIndexing(
                f"embedding failure rate {rate:.1%} sustained over the last "
                f"{SUSTAINED_WINDOW} attempts (threshold {MAX_FAILURE_RATE:.0%})"
            )


def embed_batch(texts: list[str], state: EmbedState) -> list[list[float] | None]:
    out: list[list[float] | None] = []
    for text in texts:
        vector = None
        for attempt in range(1, EMBED_RETRIES + 1):
            try:
                vector = embed_one(text)
                state.consecutive_unreachable = 0
                break
            except urllib.error.HTTPError as exc:
                # The server responded, so it is up. Treat this as a per-chunk
                # content failure (usually input too large), never as an outage
                # -- misreading it as one triggered a needless restart and a
                # spurious STOP mid-run.
                if attempt < EMBED_RETRIES:
                    time.sleep(1)
                else:
                    record_near_stop(f"embed rejected (HTTP {exc.code}): {text[:60]!r}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                state.consecutive_unreachable += 1
                if state.consecutive_unreachable >= UNREACHABLE_LIMIT:
                    if state.restarted:
                        raise StopIndexing(
                            "Ollama unreachable again after a restart already been attempted"
                        ) from exc
                    state.restarted = True
                    if not restart_ollama():
                        raise StopIndexing("Ollama restart failed; server will not come up") from exc
                    state.consecutive_unreachable = 0
                if attempt < EMBED_RETRIES:
                    time.sleep(2 ** attempt)
            except Exception as exc:  # noqa: BLE001
                if attempt < EMBED_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    record_near_stop(f"embed failed after {EMBED_RETRIES} tries: {exc}")

        state.record(vector is not None)
        out.append(vector)

    state.check_sustained_failure()
    return out


# ----------------------------------------------------------------- chroma

def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    except Exception:  # noqa: BLE001 - newer chromadb uses `configuration`
        return client.get_or_create_collection(
            name=COLLECTION, configuration={"hnsw": {"space": "cosine"}}
        )


def index_records(records: list[dict], collection, state: EmbedState, started: float) -> int:
    total = len(records)
    done = 0
    checkpoint_done = False

    for start in range(0, total, BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        vectors = embed_batch([r["text"] for r in batch], state)

        keep = [(r, v) for r, v in zip(batch, vectors) if v is not None]
        if keep:
            collection.upsert(
                ids=[r["id"] for r, _ in keep],
                embeddings=[v for _, v in keep],
                documents=[r["text"] for r, _ in keep],
                metadatas=[r["metadata"] for r, _ in keep],
            )
        done += len(batch)

        if done % 100 < BATCH_SIZE:
            write_progress(done, total, started)
            log(f"  {done}/{total} chunks ({state.failures} embed failures)")

        if free_gb() < MIN_FREE_GB:
            raise StopIndexing(f"disk free below {MIN_FREE_GB} GB")

        if not checkpoint_done and done >= CHECKPOINT_CHUNKS:
            run_quality_checkpoint(collection, records[:done], state)
            checkpoint_done = True

    return done


def run_quality_checkpoint(collection, submitted: list[dict], state: EmbedState) -> None:
    """Mandatory gate after the first 200 chunks. Any failure -> STOP."""
    import random

    log("\n=== QUALITY CHECKPOINT (first 200 chunks) ===")
    expected_ids = [r["id"] for r in submitted]
    got = collection.get(ids=expected_ids, include=["documents", "metadatas", "embeddings"])

    indexed = len(got["ids"])
    expected = len(expected_ids) - state.failures
    log(f"  chromadb has {indexed} of {expected} expected chunks")
    if indexed < expected:
        raise StopIndexing(f"chromadb count {indexed} < submitted {expected}")

    bad_dim = [
        i for i, e in zip(got["ids"], got["embeddings"])
        if len(e) != EMBED_DIM or not any(x != 0 for x in e)
    ]
    if bad_dim:
        raise StopIndexing(f"{len(bad_dim)} vectors wrong-dim or all-zero (e.g. {bad_dim[:3]})")
    log(f"  all {indexed} vectors are {EMBED_DIM}-dim and non-zero")

    missing_url = [i for i, m in zip(got["ids"], got["metadatas"]) if not m.get("source_url")]
    if missing_url:
        raise StopIndexing(f"{len(missing_url)} chunks missing source_url")
    log("  source_url present on every chunk")

    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    for rec in random.sample(submitted, min(3, len(submitted))):
        if rec["id"] not in by_id:
            continue
        doc, meta = by_id[rec["id"]]
        if doc != rec["text"]:
            raise StopIndexing(f"round-trip text mismatch on {rec['id']}")
        if meta.get("category") != rec["metadata"]["category"]:
            raise StopIndexing(f"round-trip metadata mismatch on {rec['id']}")
        log(f"  round-trip OK: {rec['id']} ({len(doc.split())} words)")

    log("=== CHECKPOINT PASSED ===\n")


def selftest() -> int:
    """Mandatory idempotency proof: index 10 chunks, rerun, count must not grow."""
    log("=== SELFTEST: deterministic-ID resumability ===")
    records = build_chunk_records()[:10]
    collection = get_collection()
    state = EmbedState()

    before = collection.count()
    index_records(records, collection, state, time.monotonic())
    after_first = collection.count()
    index_records(records, collection, state, time.monotonic())
    after_second = collection.count()

    log(f"  count before={before} after 1st={after_first} after 2nd={after_second}")
    if after_first != after_second:
        log("  FAIL: rerun duplicated chunks -- resumability is broken")
        return 1
    log("  PASS: rerun upserted in place, no duplication\n")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()

    started = time.monotonic()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log("Building chunk records...")
    records = build_chunk_records()
    log(f"  {len(records)} chunks from {MANIFEST_PATH.name}")

    collection = get_collection()
    state = EmbedState()
    log(f"  collection '{COLLECTION}' currently holds {collection.count()} chunks")

    try:
        done = index_records(records, collection, state, started)
    except StopIndexing as exc:
        log(f"\nSTOP condition: {exc}")
        record_near_stop(f"STOPPED: {exc}")
        return 1

    # Drop chunks that no longer exist. Re-extraction changes how a page splits,
    # so a page that used to make 6 chunks and now makes 4 would leave #4 and #5
    # behind -- stale text, still retrievable, still citing a real URL. Upsert
    # alone cannot remove them.
    valid = {r["id"] for r in records}
    existing = set(collection.get(include=[])["ids"])
    stale = existing - valid
    if stale:
        stale_list = sorted(stale)
        for i in range(0, len(stale_list), 500):
            collection.delete(ids=stale_list[i : i + 500])
        log(f"  removed {len(stale)} stale chunks left by re-extraction")

    elapsed = time.monotonic() - started
    log(f"\nStage B done: {done} chunks processed, {state.failures} embed failures")
    log(f"  collection now holds {collection.count()} chunks")
    log(f"  elapsed {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
