# Overnight Scrape Spec — Phase 1 full crawl

> **STATUS: COMPLETED 2026-07-26** — 1,094 pages scraped, no STOP condition.
> See `data/logs/run_summary.txt` for full results.

Read fully before executing. Work through phases in order. If anything in the
STOP conditions triggers, stop and write a report — do not improvise past it.

## Objective

Scrape the high-value content of giki.edu.pk into `data/raw/`, verified clean,
with the code (not the data) committed and pushed. This is the corpus for a
university Q&A chatbot — prioritize pages a student would ask questions about.

## Scope — what to crawl, in priority order

Seed from sitemaps directly. Do NOT crawl from the homepage or follow links.

| Priority | Sitemap(s) | ~URLs | Why |
|---|---|---|---|
| 1 | page-sitemap1–3 | 507 | Admissions, fees, academics, policies |
| 2 | department-sitemap | 9 | Department info |
| 3 | course-sitemap1–4 | 707 | Course pages |
| 4 | personnel-sitemap1–2 | 398 | Faculty/staff |

Total target: ~1,621 URLs.

**Excluded entirely (do not fetch):** post-sitemap1–9 (blog/news),
tribe_events-sitemap, category-sitemap, portfolio*, personnel_category,
course_category. If a chatbot needs news later, that's a separate phase.

## Crawl rules

- Delay: 1.5 seconds between requests (bumped from 1.0 — this is a long
  unattended run against a university server; be gentle)
- Respect robots.txt via the existing robotparser check
- Existing cert-bundle session (`build_session()`) — do not change TLS handling
- Timeout 15s per request. On failure: log URL + error to
  `data/logs/failed_urls.txt`, continue. Do NOT retry in the main pass.
- After the main pass, retry every failed URL once. Still failing → leave in log.
- Skip non-HTML content types (current behavior, keep it)
- Content quality gate: skip pages whose extracted text is under 100 words —
  log to `data/logs/skipped_thin.txt` instead of saving.
  (100, not 200 — course and personnel pages are legitimately short.)

## Output structure

Organize by source sitemap so retrieval can filter by type later:

```
data/raw/pages/<slug>.txt
data/raw/departments/<slug>.txt
data/raw/courses/<slug>.txt
data/raw/personnel/<slug>.txt
data/raw/manifest.json
data/logs/failed_urls.txt
data/logs/skipped_thin.txt
data/logs/run_summary.txt
```

manifest.json entries: url, title, slug, category (pages/departments/courses/
personnel), scraped_at, word_count.

Write the manifest INCREMENTALLY — append/flush every 25 pages, not once at
the end. A crash at page 1,400 must not lose the manifest.

## Checkpointing / resumability

Before fetching each URL, check if its output file already exists AND appears
in the manifest — if so, skip it. This makes the run resumable: if it dies at
page 900, rerunning continues from 901 rather than starting over.

## Quality checkpoint — MANDATORY, do not skip

After the FIRST 30 pages (across at least 3 categories):
1. Pause the crawl.
2. Read 6 files yourself: 2 pages, 2 courses, 1 personnel, 1 department.
3. Check each for: nav-menu leakage, empty/boilerplate content, encoding
   garbage, missing main content.
4. If 2 or more of the 6 are bad → STOP. Write findings to run_summary.txt.
   Do not crawl 1,600 pages with a broken extractor.
5. If clean → continue the full run.

## STOP conditions (halt + report, do not work around)

- More than 15% of fetches failing (server blocking us or down)
- Any HTTP 429 (rate limited) → stop immediately, note the time
- Any HTTP 403 appearing repeatedly after initial success (we got blocked)
- Disk usage of data/ exceeds 500 MB (something is wrong; text should be far
  smaller)
- The quality checkpoint fails (above)

## Git rules

- `data/` stays gitignored. NEVER commit scraped content.
- Commit code changes (scraper rewrite) with a clear message BEFORE starting
  the long run — and PUSH it. The code must survive even if this machine dies
  mid-crawl.
- After the run: update CLAUDE.md status section (mark Phase 1 scrape done,
  note counts), commit, push.

## Final report — write to data/logs/run_summary.txt

- Pages saved per category (target vs actual)
- Failed URLs count + the errors seen
- Thin pages skipped count
- Total run time
- Total corpus size on disk
- Top 5 largest files (possible junk indicators)
- Anything anomalous noticed along the way
- Recommended next step for Phase 2 (retrieval layer)

## Explicitly out of scope tonight

- PDFs (Phase 2 — docling is already in the big venv for this)
- Images / OCR
- Any retrieval/embedding/chromadb work
- Increasing scope beyond the 4 sitemap groups listed
- Any change to TLS/cert handling
