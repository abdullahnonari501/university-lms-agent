# CLAUDE.md — University LMS Agent

Read this first every session. Update it when something structural changes.

---

## Who I am

Abdullah Malik. Final-year Engineering Sciences student (Modeling, Simulation & ML),
GIK Institute, Pakistan. Graduating December 2026.

**How to talk to me:** Lead with the answer. No preamble. Short chunks, headers,
bold key actions. Tag confidence: [Certain] / [Likely] / [Guessing]. Challenge me
in the first sentence if I'm wrong — no sycophancy. One clear next action at the
end. Max one question per response. English isn't my first language and I
voice-type, so read broken grammar charitably.

**Patterns to catch and name:** goal-stacking, optimizing the system instead of
working, browsing instead of shipping.

---

## The project

A chatbot agent for university websites. Scrape the site → retrieval layer →
Qwen (via Ollama) answers student questions. Voice added later.

**Pilot:** GIKI. **Then:** replicate the pipeline for other universities.

---

## Infrastructure (settled — do not redesign)

| | Laptop (Windows) | Pop!_OS box |
|---|---|---|
| Role | Thin client. Screen only. | The actual computer |
| Runs | Cursor | Everything else |

- **Connection:** Cursor Remote Tunnel (`pop-os-tunnel`), GitHub auth, via Microsoft relay
- **Why not SSH/Tailscale:** the network runs a Fortinet firewall doing TLS
  inspection on `controlplane.tailscale.com`. Tailscale pins its cert and refuses
  the intercepted connection, hanging forever. Tunnels use the system trust store
  and pass through fine.
- **sudo is deliberately restricted** on this box — blocks `systemctl`, `chmod`,
  `chown`, `chroot`. Prefer solutions that need no root at all.
- **Durability:** GitHub. The Pop!_OS box is not permanently mine — anything not
  pushed does not exist.

**Agent roles:** Claude Code is the driver (multi-file, agentic work). Cursor is
editor + autocomplete only. Don't run both agents against the same repo.

---

## Status

**Done**
- Private repo `abdullahnonari501/university-lms-agent`
- `.gitignore` (Python + Node; excludes venv, node_modules, .env, model weights)
- Cursor tunnel connected, running as a persistent service
- Ollama + Qwen live on the Pop!_OS box
- **Phase 1 scrape complete** (2026-07-26) — **1,094 pages** of clean text from
  `giki.edu.pk`: 614 courses, 319 pages, 152 personnel, 9 departments. 6.17 MB.
  Sitemap-seeded (1,414 target URLs), 319 thin pages (<100 words) skipped,
  only 4 failures, no STOP condition. Run via `src/overnight_scrape.py`;
  see `data/logs/run_summary.txt`.
- **130 unique PDF/DOC files inventoried** in `data/logs/found_documents.txt`
  (URLs only — nothing downloaded or parsed). The file has 12,208 *lines*, but
  that's link occurrences: two footer PDFs account for 11,957 of them. Always
  `cut -f1 ... | sort -u` before quoting a count.
  "316 of 319 thin pages link to a PDF" is an artifact — for most of them the
  only links are the two site-wide footer PDFs. The `/fee/*` pages really are
  empty shells (0–8 words, no page-specific PDF).
  **But fee data DOES exist** — at `/admissions/admissions-undergraduates/
  ugrad-fees-and-expenses/` and its `/admissions-graduate/` twin, both in the
  corpus with their tables intact. An earlier note here claimed no fee schedule
  existed anywhere; that was inferred from the empty `/fee/*` pages plus no fee
  PDF, and it was simply wrong. Check every path before declaring data absent.

- **Phase 2 retrieval layer complete** (2026-07-28) — **4,210 chunks** in
  chromadb (`data/chroma`, collection `giki`, cosine), embedded with
  `nomic-embed-text` (768-dim) via Ollama. 0 embedding failures. Chunks are
  512-token target / 64 overlap; **tables preserved** (756 pages, 9,638 rows) —
  whole under 1,800 tokens, else split on row boundaries with header repeated.
  All 9 verified PDFs parsed with docling (145k words added) — handbook,
  academic calendar, both prospectuses, transport + admissions + disabilities +
  harassment policies. Query with `src/retrieve.py "..."`.
  See `data/logs/phase2_run_summary.txt`.

**Next**
- [ ] Qwen wired to retrieval
- [ ] Chat UI
- [ ] Voice

---

## Phase 1 — Scrape GIKI

**Goal:** one folder of clean text, one page per file, from `giki.edu.pk`.

1. Check `giki.edu.pk/robots.txt` — respect it, rate-limit ~1 req/sec
2. Seed on the homepage, same-domain links only, depth 2–3
3. Extract main text, strip nav/footer/scripts
4. Save `data/raw/<slug>.txt` + `manifest.json` (url, title, scraped_at)
5. **Stop and eyeball 5 random files before scaling the crawl**

**Stack:** `requests` + `beautifulsoup4`. Only reach for `playwright` if pages
turn out to be JavaScript-rendered.

**Two gotchas already paid for — don't rediscover them:**
- `giki.edu.pk` doesn't send its TLS intermediate cert. Hence the vendored
  `certs/rapidssl_intermediate.crt` + custom `build_session()`. Never "fix" this
  with `verify=False`.
- The theme (Kingster) duplicates the whole nav menu in a `<div>` **outside** any
  `<nav>`/`<header>`, so tag-stripping alone leaks it into every page. Extraction
  positively selects `#kingster-page-wrapper` instead.
- `requests`' `timeout=` bounds only the *gap between bytes*, never total elapsed
  time — a trickling server hung the crawl 15+ min mid-run, twice, with
  `timeout=15` set. `fetch()` in `overnight_scrape.py` streams under a hard
  90s wall-clock cap. Any new fetch code must go through it.
- **Never return large data through `mp.Queue.put()`.** It blocks past the OS
  pipe buffer (~64 KB) until the parent drains it — and if the parent is in
  `proc.join()`, that's a deadlock. Cost 4 debugging passes in Phase 2: every
  PDF over ~16k words "hung", every small one worked. Hand back a temp-file path.
- Pin `OMP_NUM_THREADS=1` (+ MKL/OPENBLAS) for docling, and set
  `HF_HUB_OFFLINE=1` — docling phones HuggingFace even with weights cached.
- **Fee data DOES exist** — at `/admissions/admissions-undergraduates/
  ugrad-fees-and-expenses/` and the `/admissions-graduate/` equivalent, already
  in the corpus. (An earlier note here claimed otherwise: that was inferred from
  the empty `/fee/*` pages plus no fee PDF, and it was wrong. Check every path
  before declaring data absent.)
- **HTML tables were being flattened** by `get_text()` — the fee page became
  four headers followed by four unlabelled numbers, losing which figure was
  Engineering-per-semester. `table_to_markdown()` in `scraper.py` now renders
  tables as Markdown honouring colspan *and* rowspan. Any corpus extracted
  before 2026-07-28 has flattened tables and must be re-scraped.

**Open question:** the GIKI LMS itself is behind a login. The public website is
scrapeable; the authenticated LMS is not, without institutional permission. Decide
early whether this product answers public-info questions or needs real LMS access —
it changes everything downstream.

---

## Standing reminder

Primary goal is **first freelance income by mid-August 2026**. This project is a
portfolio/product play, not an August income lever. It gets its own time block and
does not eat Upwork or Coursera time.
