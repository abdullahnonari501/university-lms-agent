# Phase 3 Spec — Wire Qwen to Retrieval (Daytime Run)

Malik is available. This spec has two kinds of boundaries:
- **AUTONOMY ZONES** — solve problems yourself, log what you did, keep moving.
- **ASK-POINTS** — stop and ask Malik. These are decisions, not obstacles.

Default posture: if a problem is *technical* (bug, limit, integration), fix it
yourself and note it. If a problem is a *product decision* (tone, behavior,
tradeoff a user would notice), ask.

## Objective

`src/answer.py`: question in → answer out, built on retrieve.py. The bot
answers in three modes and always tells the user which one it used.

## The three answer modes (core design — do not deviate)

1. **GROUNDED** — retrieved chunks answer the question. Answer from them ONLY,
   cite source_url(s). Citations must be copied verbatim from retrieved chunk
   metadata — never constructed, never remembered.
2. **GENERAL (flagged)** — retrieval finds nothing relevant AND the question is
   generic (not GIKI-specific). Answer from Qwen's general knowledge, prefixed
   with a clear visible flag, e.g.:
   "⚠ Not from GIKI's website — general knowledge:"
   General answers NEVER carry a GIKI citation.
3. **REFUSE** — retrieval finds nothing AND the question is GIKI-specific
   factual (a fee, a date, a person, a policy, a deadline). Say plainly that
   the public website doesn't cover it and suggest where to ask (admissions
   office, etc.). NEVER answer GIKI-specific facts from general knowledge —
   a plausible invented fee/date is the worst failure this bot can produce.

The GROUNDED vs GENERAL vs REFUSE decision:
- Retrieval relevance: use top-1 similarity score with a threshold you tune
  empirically (see Autonomy). Above threshold → GROUNDED.
- Below threshold → classify the question: GIKI-specific factual → REFUSE;
  generic → GENERAL with flag. When classification is genuinely ambiguous,
  prefer REFUSE (safe) over GENERAL (risky).

## Build order

### Stage 1 — Prompt + pipeline draft
1. answer.py: takes a question, calls retrieve.py (top-k=5), builds the
   prompt (system rules + chunks with their source_urls + question), calls
   Qwen via Ollama, returns answer + mode + citations.
2. Mind the context window: verify the *served* context limit of the Qwen
   model on this box (ollama show / a live probe — trust the server, not the
   model card; we've been burned twice). If 5 chunks + system + question
   overflow, trim to fewer chunks by score. Log the measured limit.
3. Run these 3 questions and capture full output:
   - "What are the undergraduate fee charges?" (GROUNDED — the table)
   - "What is a GPA and how is it calculated?" (GENERAL — flagged)
   - "What is the hostel fee for international students?" (REFUSE if corpus
     lacks it — verify first, don't assume)

**ASK-POINT 1:** Show Malik the 3 outputs (full text, mode, citations) BEFORE
any further building. This is a product-tone review: does the answer style,
flag wording, and refusal wording feel right? Do not proceed past this
without his sign-off.

### Stage 2 — Model head-to-head
1. Pull text-only qwen2.5:7b (vision model stays — do not delete it).
2. Run the SAME 6 questions (3 above + 3 more you pick spanning courses,
   faculty, policies) through both models on identical retrieved chunks.
3. Record per model: answer quality notes, response latency, VRAM footprint.

**ASK-POINT 2:** Present the comparison table. Malik picks the model. Do not
pick for him — this is his call between speed/quality/future-vision-use.

### Stage 3 — Validation battery (after model is chosen)
1. 15 questions: 8 GROUNDED-expected (fees, calendar, courses incl. a CLO
   table question, faculty, transport, discipline, scholarships, admissions),
   4 GENERAL-expected (generic academic/life questions), 3 REFUSE-expected
   (hostel fee intl, "my grades", a fabricated-sounding GIKI specific).
2. For each: log question, mode chosen, answer, citations, latency to
   data/logs/phase3_validation.txt.
3. Automatic checks: every GROUNDED answer has ≥1 citation present in its
   retrieved chunks; every GENERAL answer carries the flag and zero GIKI
   citations; every REFUSE names where to ask instead.
4. Mode-accuracy: if more than 2 of 15 land in the wrong mode, stop and
   tune the threshold/classifier, rerun. If still failing after 2 tuning
   rounds → ASK-POINT (bring the failures to Malik, don't keep grinding).

## AUTONOMY ZONES (fix it yourself, log it, move on)

- Ollama quirks, context limits, token counting — measure the live server,
  adapt, note discrepancies vs docs.
- Retrieval integration bugs, k-value adjustments, chunk trimming logic.
- Relevance threshold tuning against the validation set.
- The known nomic search_query:/search_document: prefix issue: if weak
  retrieval blocks validation (EE-teachers and transport questions), you MAY
  apply the prefix convention and re-index (~5 min, idempotent) without
  asking — but report before/after retrieval scores for both weak queries.
- Prompt wording iterations to fix concrete failures (hallucinated citation,
  ignored context, verbosity) — iterate freely, keep the three-mode contract.
- Any crash/restart of Ollama: restart once, resume; twice → tell Malik.

## HARD RULES (no autonomy)

- Never let a GIKI-specific fact be answered from general knowledge.
- Never fabricate or alter a source_url.
- Never delete models, corpus files, or the chroma index.
- Never change the three-mode contract or flag visibility without asking.
- data/ stays gitignored; commit + push code at each stage boundary.

## Report — data/logs/phase3_run_summary.txt

- Measured context limit + any docs-vs-server discrepancies
- Head-to-head table (kept for the record even after Malik's pick)
- Validation: 15 questions, modes, accuracy, latencies
- Threshold value chosen and how it was tuned
- Prefix fix applied or not, with before/after scores if applied
- Update CLAUDE.md (Phase 3 state + new gotchas), commit, push

## OUT of scope

- Chat UI, voice, streaming — Phase 4
- Ingesting more PDFs or images
- Conversation memory / multi-turn — single-question answering only for now
