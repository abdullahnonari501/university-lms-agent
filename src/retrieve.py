"""Phase 2 Stage C: query the chromadb index built by build_index.py.

Embeds the query with the same model used for indexing (nomic-embed-text) and
returns top-k chunks with score, source_url and title.

Queries are embedded with nomic-embed-text's "search_query: " prefix, matching
the "search_document: " prefix build_index.py applies to chunks. The two must
always change together: a mismatch silently degrades every result.

Usage:
    python3 src/retrieve.py "when does the fall semester start?"
    python3 src/retrieve.py --test-battery
"""

import json
import math
import random
import re
import sys
from collections import Counter
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
# Must match the prefix build_index.py uses for documents (see DOC_PREFIX there).
QUERY_PREFIX = "search_query: "
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
    payload = json.dumps({"model": EMBED_MODEL, "prompt": f"{QUERY_PREFIX}{text}"}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["embedding"]


def get_collection():
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)


_LEXICAL: dict | None = None
TOKEN_RE = re.compile(r"[a-z0-9]+")
CANDIDATES = 60          # per retriever, before fusion
RRF_K = 60               # standard reciprocal-rank-fusion constant
# Weighted RRF: a hit in an intent-routed pool counts for more. Unweighted, a
# globally-retrieved chunk scores twice (dense list + BM25 list) while a routed
# hit scores once per routed list, so the routing signal is diluted exactly when
# it is most informative.
ROUTED_WEIGHT = 2.5


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _lexical_index(collection) -> dict:
    """BM25 over every chunk, built once per process and cached.

    Dense retrieval alone cannot do entity lookup: "Who is the Dean of FCSE?"
    is semantically nearest the long FCSE overview prose, while the answer sits
    in a 49-word contact block whose distinguishing feature is that it literally
    contains "DEAN" and "FCSE". Embeddings smooth exactly that signal away;
    term matching preserves it.
    """
    global _LEXICAL
    if _LEXICAL is None:
        got = collection.get(include=["documents", "metadatas"])
        tokens = [_tokenize(d) for d in got["documents"]]
        df: Counter[str] = Counter()
        for toks in tokens:
            df.update(set(toks))
        n = max(1, len(tokens))
        _LEXICAL = {
            "ids": got["ids"],
            "documents": got["documents"],
            "metadatas": got["metadatas"],
            "tf": [Counter(t) for t in tokens],
            "lengths": [len(t) for t in tokens],
            "df": df,
            "n": n,
            "avgdl": sum(len(t) for t in tokens) / n,
        }
    return _LEXICAL


def _bm25_top(collection, query: str, limit: int, k1: float = 1.5, b: float = 0.75):
    idx = _lexical_index(collection)
    terms = [t for t in _tokenize(query) if len(t) > 1]
    if not terms:
        return []
    scores: dict[int, float] = {}
    for term in set(terms):
        df = idx["df"].get(term, 0)
        if df == 0:
            continue
        idf = math.log(1 + (idx["n"] - df + 0.5) / (df + 0.5))
        for i, tf_counter in enumerate(idx["tf"]):
            tf = tf_counter.get(term, 0)
            if not tf:
                continue
            norm = 1 - b + b * (idx["lengths"][i] / idx["avgdl"])
            scores[i] = scores.get(i, 0.0) + idf * (tf * (k1 + 1)) / (tf + k1 * norm)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
    return [(idx["ids"][i], idx["documents"][i], idx["metadatas"][i], s) for i, s in ranked]


PERSON_INTENT_RE = re.compile(
    r"\b(who|whose|dean|head of|chairman|chairperson|professor|teaches|taught"
    r"|supervisor|advisor|adviser|faculty member|staff)\b",
    re.IGNORECASE,
)


def _route_categories(query: str) -> list[str]:
    """Categories worth searching in their own right for this query.

    A 49-word staff profile cannot out-rank prospectus prose in a pool of 4,363
    mixed chunks, however the scoring is tuned -- "dean" is not even rare, since
    every faculty page carries "MESSAGE FROM THE DEAN". Searching personnel as
    its own pool for person-shaped questions removes the volume mismatch instead
    of trying to out-tune it.
    """
    return ["personnel"] if PERSON_INTENT_RE.search(query) else []


def search(query: str, k: int = DEFAULT_K, category: str | None = None) -> list[dict]:
    """Hybrid retrieval: dense semantic + BM25 lexical, fused by reciprocal rank.

    Reciprocal-rank fusion is used rather than blending raw scores because
    cosine similarity and BM25 live on incomparable scales; ranks do not.

    `score` stays the dense cosine similarity so the GROUNDED/REFUSE threshold
    keeps its meaning. Chunks surfaced only lexically report the dense score of
    the best dense hit for the same source, or 0.0 -- fusion widens what the
    model gets to read, it does not inflate the relevance signal.
    """
    collection = get_collection()
    where = {"category": category} if category else None

    dense = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=CANDIDATES,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    dense_rows = list(zip(dense["ids"][0], dense["documents"][0],
                          dense["metadatas"][0], dense["distances"][0]))
    dense_score = {cid: 1.0 - dist for cid, _, _, dist in dense_rows}

    lexical_rows = _bm25_top(collection, query, CANDIDATES)
    if category:
        lexical_rows = [r for r in lexical_rows if r[2].get("category") == category]

    fused: dict[str, float] = {}
    payload: dict[str, tuple] = {}
    for rank, (cid, doc, meta, _dist) in enumerate(dense_rows, 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        payload[cid] = (doc, meta)
    for rank, (cid, doc, meta, _s) in enumerate(lexical_rows, 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        payload.setdefault(cid, (doc, meta))

    # Intent-routed pools, searched separately so a small category is not
    # drowned by a large one's chunk count.
    for routed in ([] if category else _route_categories(query)):
        pool_n = max(10, k * 2)
        sub = collection.query(
            query_embeddings=[embed_query(query)],
            n_results=pool_n,
            where={"category": routed},
            include=["documents", "metadatas", "distances"],
        )
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(sub["ids"][0], sub["documents"][0], sub["metadatas"][0],
                sub["distances"][0]), 1
        ):
            fused[cid] = fused.get(cid, 0.0) + ROUTED_WEIGHT / (RRF_K + rank)
            payload.setdefault(cid, (doc, meta))
            dense_score.setdefault(cid, 1.0 - dist)

        routed_lex = [r for r in _bm25_top(collection, query, CANDIDATES)
                      if r[2].get("category") == routed][:pool_n]
        for rank, (cid, doc, meta, _s) in enumerate(routed_lex, 1):
            fused[cid] = fused.get(cid, 0.0) + ROUTED_WEIGHT / (RRF_K + rank)
            payload.setdefault(cid, (doc, meta))

    hits = []
    for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:k]:
        doc, meta = payload[cid]
        hits.append({
            "text": doc,
            "score": dense_score.get(cid, 0.0),
            "source_url": meta.get("source_url", ""),
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "doc_type": meta.get("doc_type", ""),
            "source_date": meta.get("source_date", ""),
            "date_basis": meta.get("date_basis", ""),
        })
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
