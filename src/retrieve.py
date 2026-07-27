"""Phase 2 Stage C: query the chromadb index built by build_index.py.

Embeds the query with the same model used for indexing (nomic-embed-text) and
returns top-k chunks with score, source_url and title.

Note: nomic-embed-text documents a "search_query:" / "search_document:" prefix
convention that typically improves retrieval. It is deliberately NOT used here
-- the spec fixed the embedding strategy, and index and query must match. It is
flagged in phase2_run_summary.txt as a daytime decision, not a mid-run change.

Usage:
    python3 src/retrieve.py "when does the fall semester start?"
    python3 src/retrieve.py --test-battery
"""

import json
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = REPO_ROOT / "data" / "chroma"
RAW_DIR = REPO_ROOT / "data" / "raw"
LOGS_DIR = REPO_ROOT / "data" / "logs"
TEST_OUTPUT = LOGS_DIR / "phase2_test_queries.txt"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "giki"
DEFAULT_K = 5

MIN_EXPECTED_CHUNKS = 2500
MAX_EXPECTED_CHUNKS = 6000

TEST_QUERIES = [
    "What are the admission requirements for undergraduate programs?",
    "When does the fall semester start?",
    "What documents are needed for admission?",
    "Tell me about the Faculty of Computer Science",
    "Who teaches in the Electrical Engineering department?",
    "What is the hostel/accommodation policy?",
    "How does the GIKI transport system work?",
    "What are the rules about student discipline?",
    "What scholarships are available?",
    "What courses are in the FCSE program?",
]


def embed_query(text: str) -> list[float]:
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["embedding"]


def get_collection():
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)


def search(query: str, k: int = DEFAULT_K, category: str | None = None) -> list[dict]:
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        where={"category": category} if category else None,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "score": 1.0 - dist,  # cosine distance -> similarity
                "source_url": meta.get("source_url", ""),
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "doc_type": meta.get("doc_type", ""),
            }
        )
    return hits


def sanity_checks(collection) -> list[str]:
    """Index-level checks required by the spec. Returns human-readable lines."""
    lines = []
    total = collection.count()
    lines.append(f"Total chunks indexed: {total}")
    if not (MIN_EXPECTED_CHUNKS <= total <= MAX_EXPECTED_CHUNKS):
        lines.append(
            f"  ** ANOMALY: outside the expected {MIN_EXPECTED_CHUNKS}-{MAX_EXPECTED_CHUNKS} range **"
        )

    sample = collection.get(limit=min(total, 20000), include=["metadatas"])
    per_category: dict[str, int] = {}
    per_type: dict[str, int] = {}
    for meta in sample["metadatas"]:
        per_category[meta.get("category", "?")] = per_category.get(meta.get("category", "?"), 0) + 1
        per_type[meta.get("doc_type", "?")] = per_type.get(meta.get("doc_type", "?"), 0) + 1
    lines.append("Chunks per category:")
    for cat, n in sorted(per_category.items()):
        lines.append(f"  {cat:12} {n}")
    lines.append(f"Chunks per doc_type: {dict(sorted(per_type.items()))}")

    # 10 random chunks must match their source file on disk
    ids = sample["ids"]
    picked = random.sample(ids, min(10, len(ids)))
    got = collection.get(ids=picked, include=["documents", "metadatas"])
    mismatches = 0
    for doc, meta in zip(got["documents"], got["metadatas"]):
        source = REPO_ROOT / meta.get("source_file", "")
        if not source.exists():
            mismatches += 1
            continue
        haystack = source.read_text(encoding="utf-8")
        probe = " ".join(doc.split()[:12])
        if probe and probe not in " ".join(haystack.split()):
            mismatches += 1
    lines.append(f"Random-chunk source verification: {10 - mismatches}/10 matched their file on disk")
    if mismatches:
        lines.append(f"  ** {mismatches} chunk(s) did not match source **")
    return lines


def run_test_battery() -> int:
    collection = get_collection()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    out = [
        "Phase 2 retrieval test battery",
        f"Run at: {datetime.now(timezone.utc).isoformat()}",
        f"Embedding model: {EMBED_MODEL}  |  collection: {COLLECTION}  |  top-k shown: 3",
        "=" * 78,
        "",
    ]

    for n, query in enumerate(TEST_QUERIES, 1):
        out.append(f"[{n}] {query}")
        try:
            hits = search(query, k=3)
        except Exception as exc:  # noqa: BLE001
            out.append(f"    QUERY FAILED: {type(exc).__name__}: {exc}")
            out.append("")
            continue

        if not hits:
            out.append("    (no results)")
        for rank, hit in enumerate(hits, 1):
            snippet = " ".join(hit["text"].split())[:240]
            out.append(f"  {rank}. score={hit['score']:.4f}  [{hit['category']}/{hit['doc_type']}]")
            out.append(f"     title : {hit['title'][:90]}")
            out.append(f"     source: {hit['source_url']}")
            out.append(f"     text  : {snippet}...")
        out.append("")
        print(f"[{n}/10] {query[:60]} -> {len(hits)} hits", flush=True)

    out.append("=" * 78)
    out.append("INDEX SANITY CHECKS")
    out.extend(sanity_checks(collection))

    TEST_OUTPUT.write_text("\n".join(out), encoding="utf-8")
    print(f"\nWrote {TEST_OUTPUT}", flush=True)
    print("\n".join(out[-14:]), flush=True)
    return 0


def main() -> int:
    if "--test-battery" in sys.argv:
        return run_test_battery()

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 1

    for hit in search(" ".join(args)):
        print(f"\n[{hit['score']:.4f}] {hit['title']}  ({hit['category']})")
        print(f"  {hit['source_url']}")
        print(f"  {' '.join(hit['text'].split())[:300]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
