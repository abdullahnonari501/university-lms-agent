# Overnight Spec — Phase 2: Retrieval Layer

Read fully before executing. Work phases in order. If a STOP condition
triggers, stop and write the report — do not improvise past it.

## Objective

Turn the Phase 1 corpus (1,094 HTML pages + 9 real PDFs) into a working,
queryable retrieval layer: chunked, embedded with nomic-embed-text via Ollama,
stored in chromadb, validated with test queries. NO Qwen wiring, NO chat UI
tonight — retrieval only. Quality gets judged by Malik tomorrow.

## Decisions already made (do not re-litigate)

- Embedding model: **nomic-embed-text via Ollama** (768-dim)
- Chunking: **512 tokens target, 64-token overlap, split on paragraph/heading
  boundaries** — 71% of pages stay single-chunk by design
- Storage: chromadb `PersistentClient(path="data/chroma")`, collection
  `giki`, `hnsw:space = "cosine"`
- Metadata per chunk: source_url, title, category, doc_type (html|pdf),
  chunk_index, n_chunks, source_file, scraped_at
- PDFs: ONLY the 9 verified to exist (handbook, academic calendar 2025-26,
  transportation policy, UG admissions policy, FCSE prospectus breakdown,
  graduate prospectus 2024, UG prospectus 2021, + the other two from the
  verified inventory). The fee schedule / degree-requirements / clearance
  PDFs DO NOT EXIST — do not search for them again.
- Environment: run Phase 2 from **.venv** (has docling + chromadb).
  .lms-venv stays untouched for the scraper.

## Phase 0 — Pre-flight (all must pass before any long work)

1. `git status` clean check: commit + push any uncommitted code FIRST.
   Nothing tonight starts until the repo state is safe on GitHub.
2. Verify .venv imports: `docling`, `chromadb` — importable without error.
3. Ollama server: check if running. If not, start it as a background process
   that survives this shell (nohup or setsid — NOT tied to the session).
   Remember systemctl under sudo is blocked on this box; user-level only.
4. `ollama pull nomic-embed-text`, then verify with one real embedding call:
   confirm it returns a 768-dim vector.
5. Confirm qwen model is present in `ollama list` (not used tonight, just
   confirm it survived the reboot — note result in report).
6. Corpus integrity: file count in data/raw/** matches manifest count (1,094).
   Mismatch > 5 files → STOP (corpus incomplete, do not index a broken corpus).
7. Disk free space > 5 GB. Below → STOP.

## Stage A — PDF ingestion (src/ingest_docs.py)

1. Download the 9 PDFs into `data/docs/` using the existing hardened fetch
   (90s wall-clock cap, cert bundle session). Skip any already downloaded
   with matching size.
2. Parse each with docling → plain text into `data/raw/documents/<slug>.txt`
   + append entries to the manifest (category: "documents", doc_type: "pdf").
3. Per-PDF guard: wrap each parse in its own try/except AND a hard timeout
   of 10 minutes per document (docling can hang on malformed PDFs). On
   failure or timeout: log to `data/logs/phase2_failed_docs.txt`, continue.
4. Per-PDF validation: extracted text must be ≥ 200 words. Below → treat as
   parse failure, log, continue.
5. STOP condition: if ALL 9 PDFs fail, the environment is broken — stop.

## Stage B — Chunk + embed + index (src/build_index.py)

1. Chunk all HTML pages + parsed documents per the decided strategy.
2. **Deterministic chunk IDs**: `{category}/{slug}#{chunk_index}`. This makes
   indexing idempotent — a rerun after a crash or power cut upserts the same
   IDs instead of duplicating. This is the resumability mechanism; do not
   use random IDs.
3. Embed in batches (e.g. 32 chunks per call pattern). On an Ollama call
   failure: retry 3x with backoff. If Ollama is unreachable 5 consecutive
   times: attempt ONE restart of the server, resume; if it dies again → STOP.
4. Progress log: append to `data/logs/phase2_progress.txt` every 100 chunks
   (count done / total, elapsed). Unbuffered writes — we learned this the
   hard way with the silent scrape deaths.
5. Quality checkpoint after the FIRST 200 chunks — MANDATORY:
   - every vector is 768-dim, non-zero
   - chromadb count matches chunks submitted
   - pull 3 random chunks back out by ID and confirm text + metadata intact
   - source_url present on every sampled chunk (non-negotiable — the bot
     must cite sources)
   Any check fails → STOP. Do not embed 4,000 chunks on a broken pipeline.
6. STOP conditions: >5% of chunks failing to embed; disk free < 2 GB.

## Stage C — Retrieval + validation (src/retrieve.py)

1. Build retrieve.py: query → embed query → top-k=5 from chromadb, optional
   category filter, returns text + score + source_url + title.
2. Run this fixed battery of 10 test queries and save FULL results (top-3
   per query with scores and source URLs) to `data/logs/phase2_test_queries.txt`:
   - "What are the admission requirements for undergraduate programs?"
   - "When does the fall semester start?"
   - "What documents are needed for admission?"
   - "Tell me about the Faculty of Computer Science"
   - "Who teaches in the Electrical Engineering department?"
   - "What is the hostel/accommodation policy?"
   - "How does the GIKI transport system work?"
   - "What are the rules about student discipline?"
   - "What scholarships are available?"
   - "What courses are in the FCSE program?"
3. Sanity checks on the final index:
   - total chunk count within 2,500–6,000 (outside → something went wrong,
     note it prominently)
   - counts per category recorded
   - 10 random chunks sampled: text matches its source file on disk
4. Do NOT judge answer quality yourself beyond "returned something plausibly
   on-topic vs returned garbage" — Malik reviews quality tomorrow.

## Failure & resume rules

- Any crash or power cut: rerunning build_index.py must resume via the
  deterministic IDs, not duplicate. Verify this logic exists before the
  long run (index 10 chunks, rerun, confirm count stays 10 — this test is
  mandatory in Stage B before full indexing).
- All logs under data/logs/. All timestamps UTC.
- Never delete or modify data/raw/** — it is the Phase 1 output. Read-only
  tonight except adding data/raw/documents/.

## Git rules

- Before the long run: commit + push ingest_docs.py, build_index.py,
  retrieve.py, and the updated spec.
- data/ stays gitignored (including data/chroma and data/docs).
- After the run: update CLAUDE.md status (Phase 2 state, chunk counts,
  anything learned worth a "gotcha" entry), commit, push.

## Final report — data/logs/phase2_run_summary.txt

- Pre-flight results (incl. whether qwen survived the reboot)
- PDFs: downloaded / parsed / failed, word counts each
- Chunks: total, per category, embedding failures
- Timings per stage
- The 10 test queries with top-3 results each (also in the dedicated file)
- Anything anomalous
- Recommended next step for Phase 3 (Qwen wiring)

## Explicitly OUT of scope tonight

- Wiring Qwen / answer generation
- Chat UI, voice
- The other ~120 PDFs
- Re-scraping anything
- Changing chunking strategy mid-run because results "look off" — record
  observations in the report instead; strategy changes are a daytime decision
