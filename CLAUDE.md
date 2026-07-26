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
  **316 of the 319 thin pages link to a PDF** — the academic calendar, fee
  schedules and clearance rules live in those docs, not in HTML. Phase 2 must
  ingest them or the bot can't answer the most common student questions.

**Next**
- [ ] Embeddings / retrieval layer
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

**Open question:** the GIKI LMS itself is behind a login. The public website is
scrapeable; the authenticated LMS is not, without institutional permission. Decide
early whether this product answers public-info questions or needs real LMS access —
it changes everything downstream.

---

## Standing reminder

Primary goal is **first freelance income by mid-August 2026**. This project is a
portfolio/product play, not an August income lever. It gets its own time block and
does not eat Upwork or Coursera time.
