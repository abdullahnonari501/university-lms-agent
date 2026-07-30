# Night Run Report — 2026-07-30

Three tasks, all complete. README text is at the bottom for review before it
goes into `README.md` — everything else is committed.

---

## 1. Voice from your laptop

### The URL to open

**First choice — VS Code forwarded port (no cert warning):**

In VS Code → **PORTS** panel → **Add Port** → `8137` → then open

```
http://localhost:8137
```

**on your laptop.** A forwarded port arrives as `localhost` on your machine, and
browsers treat `localhost` as a secure context regardless of scheme — so the
microphone works with no certificate involved. Visibility can stay **Private**;
this path does not need Public.

**Fallback — direct HTTPS (works today, one warning):**

```
https://10.1.131.37:8443
```

Click through *Advanced → Proceed*. Only works if your laptop is on the same
network as the box.

### Which path I took, in one line

I made the forwarded-port path actually possible by fixing why VS Code refused
it, and kept HTTPS on 8443 as a verified fallback — the fix was a socket
binding bug, not anything to do with Whisper.

### The real blocker, and it was mine

"Unable to forward localhost:8137" was not a stale registration. Your screenshot
proved that: neither 8000 nor 8137 appeared in the PORTS list at all.

Two separate address-family problems, in sequence:

1. The server bound IPv4-only. This box's `/etc/hosts` resolves `localhost` to
   `::1` **before** `127.0.0.1`, so anything connecting by name found nothing
   listening. Evidence: `[::1]:8137` returned nothing while `127.0.0.1:8137`
   returned 200.
2. My fix for that — a dual-stack IPv6 socket — then made it **worse in a new
   way**. VS Code's port scanner reads `/proc/net/tcp`, the IPv4 table, and a
   dual-stack IPv6 socket appears only in `/proc/net/tcp6`. Every port VS Code
   *did* auto-forward here (ollama 11434, extension host 14302) is listed in the
   IPv4 table; mine was in neither place it looked.

Final shape: one listener per family on the same port, IPv6 with `V6ONLY=1` so
they coexist. Port 8137 now appears in **both** kernel tables, and all four
access paths return 200 — `127.0.0.1`, `[::1]`, `localhost` by name, and the LAN
address.

Worth knowing: the reason your port never auto-forwarded in the first place is
that I start the server with `nohup`, outside a VS Code terminal. VS Code only
auto-detects ports from its own terminals, which is why **Add Port** is needed.

### Full loop tested

Piper renders a spoken question → `/transcribe` → `/ask` → `/speak`, over HTTPS:

| Step | Result |
|---|---|
| 1. mic audio (simulated by Piper) | 98,464 bytes WAV |
| 2. Whisper heard | `"What are the undergraduate fee charges?"` — exact, 9.5s |
| 3. answer | **GROUNDED**, 3 citations, 13.4s |
| 4. Piper spoke the reply | 2.3 MB WAV, 2.0s |
| 5. voice follow-up | heard `"What about for MS students?"`, rewritten to `"What are the fee charges for MS students?"`, answered GROUNDED |

Step 5 is the part worth noting: **memory works through voice**, not just typing.

One caveat I cannot test from here: I have no microphone on this box, so steps
1–2 used Piper's audio as a stand-in for yours. Real speech is messier than
synthesised speech — expect accuracy to be somewhat lower than that exact match
suggests, though `STT_VOCAB` biasing is in place for GIKI/FCSE/CGPA.

---

## 2. Repo presentability

**Done and committed:**

- `requirements.txt` was actively wrong — it listed only `requests` and
  `beautifulsoup4` while the project needs `docling`, `chromadb`, `transformers`
  and `torch`. Rewritten, grouped by phase, with the non-pip runtime
  dependencies (ollama, piper) called out explicitly.
- `CLAUDE.md` synced to true state: Phase 1 was still claiming 1,094 pages
  (now 1,306 files), Phase 2 still claimed 4,210 chunks (now 4,363), Phase 4
  still pointed at port 8000. Added the off-box backup location, and two hard-won
  gotchas: microphones need a secure context, and both address families must be
  bound.
- Removed dead files: `data/_raw_backup_preRefilter` (8.7 MB — the *pre-table-fix*
  corpus, superseded and now safely covered by the off-box backup) and
  `models/piper.tar.gz` (26 MB — already unpacked).

**Deliberately kept:** `src/compare_models.py`, `src/refilter_preview.py`,
`src/coverage_check.py`, `src/validate.py`. These look like one-offs but each
produced results cited in a report and each is re-runnable evidence, not scratch
work. The three `*_SPEC.md` files are yours and record what was actually asked
for.

**Not committed, awaiting your review:** the README text below.

---

## 3. Data backup

| | |
|---|---|
| Archive | `giki-data-20260730.tar.gz` |
| Size | **224 MB** (234.5 MB as uploaded) |
| Landed | GitHub Release `data-backup-20260730` |
| URL | https://github.com/abdullahnonari501/university-lms-agent/releases/tag/data-backup-20260730 |
| State | `uploaded`, confirmed via `gh release view` |

Verified **before** uploading, not after: `tar tzf` lists 2,525 entries including
`data/raw/manifest.json`, 1,306 `.txt` files, and the chroma index. An archive
you have not listed is not a backup.

Restore after a fresh clone:

```bash
tar xzf giki-data-20260730.tar.gz -C /path/to/repo
```

Breakdown of what is in it: `data/chroma` 131 MB (rebuildable in ~7 min from
raw), `data/docs` 161 MB (re-downloadable), `data/raw` 9.4 MB (**the expensive
part — ~2 hours to re-scrape**), `data/logs` 4.1 MB.

Note it compressed 314 MB → 224 MB largely because the PDFs do not compress; if
you ever want a smaller routine backup, `data/raw` + `data/logs` alone is ~14 MB
and covers everything that is genuinely slow to rebuild.

---

## Left running

- HTTPS `:8443` and HTTP `:8137`, both dual-family, both with voice available
- Ollama with qwen2.5:7b, nomic-embed-text, qwen2.5vl
- Nothing survives a reboot — all `nohup` processes

## Open items, unchanged

1. Streaming replies — answers take 5–18s behind a typing dot
2. UG/graduate fee page scoping
3. Reboot persistence for ollama + server

---
---

# PROPOSED README.md — review before it becomes final

```markdown
# GIKI Assistant

A self-hosted question-answering assistant for a university's public website.
Scrapes the site, indexes it, and answers student questions **only** from what it
can cite — with voice in and out. Everything runs locally: no API keys, no
per-query cost, no data leaving the machine.

Pilot institution: **GIK Institute of Engineering Sciences and Technology**
(`giki.edu.pk`). The pipeline is meant to be re-pointed at other universities.

---

## Why it exists

Students ask the same questions every intake — fees, deadlines, course
prerequisites, hostel and clearance rules — and the answers are buried across
1,300 pages and a dozen PDFs. A general chatbot answers those questions
confidently and often wrongly. This one refuses instead.

The design constraint that shaped everything: **a plausible invented fee or
deadline is worse than no answer.**

---

## What it does

Ask a question, get one of three clearly-labelled answers:

| Mode | When | Guarantee |
|---|---|---|
| **GROUNDED** | The corpus answers it | Answer comes only from retrieved text, with source URLs |
| **GENERAL** | Question is generic, corpus silent | Answered from the model's own knowledge, behind a visible ⚠ flag, never with a GIKI citation |
| **REFUSE** | Question is institution-specific, corpus silent | Says so plainly and names who to ask |

Plus: multi-turn memory, Markdown tables preserved end to end, and local
speech-to-text and text-to-speech.

---

## Architecture

```
giki.edu.pk
    │  sitemap-seeded crawl, hardened fetch          src/overnight_scrape.py
    │  identity-based junk filter, table-preserving  src/scraper.py
    ▼
data/raw/  1,297 pages + 9 parsed PDFs               src/ingest_docs.py (docling)
    │
    │  512-token chunks, tables kept whole,
    │  source dates extracted                        src/build_index.py
    ▼
data/chroma/  4,363 dated chunks (nomic-embed-text, 768-dim)
    │
    │  dense + BM25 fused by reciprocal rank,
    │  query-intent routing, LLM rerank              src/retrieve.py
    ▼
three-mode answering, fabrication + staleness guards src/answer.py  (qwen2.5:7b)
    │
    ▼
chat UI, multi-turn, voice                           src/serve.py + src/voice.py
```

### Retrieval is hybrid, for a reason

Dense embeddings alone could not answer *"Who is the Dean of FCSE?"* — the
question is semantically nearest long faculty-overview prose, while the answer sat
in a 49-word contact block whose distinguishing feature was the literal word
"DEAN". Embeddings smooth exactly that signal away.

What fixed it: BM25 fused with dense retrieval by reciprocal rank (their scores
are on incomparable scales; their ranks are not), person-shaped queries routed to
search the personnel category as its own pool, a cap of 3 chunks per source page,
then an LLM rerank of the pool. The Dean's page went from **outside the top 40 to
rank 1**; a faculty-listing query went from 11th to 1st.

### Two guards, because grounding is not enough

**Fabrication.** Both candidate models invented a Dean's name — "Dr. Wahid Haqim"
and "Dr. Umar Haq" — and returned it as GROUNDED *with citations*, the most
convincing shape a wrong answer can take. `unsupported_claims()` verifies every
titled name and 4+ digit figure against the evidence actually shown to the model;
anything absent routes to a refusal.

**Staleness.** A live faculty page reads "Dean FCSE, September 2019 to August
2023". The model asserted that person as current while quoting the end date in the
same sentence. Chunks now carry a source date (year in filename, upload path, or
*undated* meaning current), each prompt source shows a currency line, and
`ended_terms()` injects an explicit warning where a role's term has already
passed. Three prompt rewrites had failed at this; putting the signal in the data
worked immediately.

---

## Running it

Requires [Ollama](https://ollama.com) and Python 3.10+.

```bash
# models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# python deps (see requirements.txt for what belongs where)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`data/` is gitignored. Either restore the published snapshot:

```bash
# from the latest data-backup-* release
tar xzf giki-data-*.tar.gz -C .
```

…or rebuild from scratch (~2 hours):

```bash
.venv/bin/python3 src/overnight_scrape.py     # crawl
.venv/bin/python3 src/ingest_docs.py          # download + parse PDFs
.venv/bin/python3 src/build_index.py          # chunk, embed, index
```

Then ask, on the command line or in a browser:

```bash
.venv/bin/python3 src/answer.py "what are the undergraduate fee charges?"
.venv/bin/python3 src/serve.py --host 0.0.0.0 --port 8443
```

**Voice needs a secure context.** Browsers block microphone access on plain HTTP
except on `localhost`. Use HTTPS (a self-signed cert is generated into `certs/`),
or reach the page through a forwarded port so it arrives as `localhost`.

For text-to-speech, unpack [Piper](https://github.com/rhasspy/piper/releases) and
a voice into `models/`.

---

## Validation

50 questions across the three modes: **44/50 correct mode, 0 contract failures**
(every GROUNDED answer cites retrieved evidence; every GENERAL answer carries its
flag and no GIKI citation; every REFUSE names somewhere to ask).

Four of the six "misses" are the corpus correctly beating generic knowledge — for
*"what does a credit hour mean"* it returns GIKI's own rule rather than a
textbook definition. Judged on behaviour rather than my prior expectations:
**48/50**. See `data/logs/phase3_validation.txt`.

---

## Engineering notes

Problems that cost real time, kept here so they are not rediscovered:

- **`requests`' `timeout=` bounds the gap between bytes, not total elapsed time.**
  A trickling server hung the crawl 15+ minutes twice with `timeout=15` set. All
  fetching goes through a hard wall-clock cap.
- **`mp.Queue.put()` blocks past the OS pipe buffer (~64 KB).** With the parent in
  `proc.join()`, that deadlocks. Every PDF over ~16k words "hung"; every small one
  worked. Cost four debugging passes chasing OCR, fork semantics and network
  before the size correlation became obvious. Workers hand back a temp-file path.
- **Trust the server, not the model card.** `ollama show` reported a 128k context;
  the server allocated 4,096. `nomic-embed-text` is served with a 2,048-token slot
  against a card claiming 8,192 — oversized chunks were *rejected outright*, not
  truncated, so they were silently missing from the index.
- **Similarity scores do not separate the answer modes.** Measured across 50
  questions the distributions overlap almost entirely (lowest GROUNDED 0.666 <
  highest REFUSE 0.774). No threshold works; the decision is made by reading the
  evidence, not scoring it.
- **Never substring-match names or figures.** "haq" matches inside "Ishaq" —
  present in nearly every chunk via "Ghulam Ishaq Khan Institute" — which let a
  fabricated name pass. "75,000" matches inside "175,000".
- **Word count is a bad proxy for value.** A 100-word floor silently deleted the
  Dean's 49-word page, the 3-word academic calendar and the 8-word clearance page.
  Junk is now excluded by identity (known theme-demo URLs) rather than length,
  which recovered 158 real pages.
- **HTML tables must survive extraction.** `get_text()` flattened the fee table
  into four headers followed by four unlabelled numbers, losing which figure was
  per-semester. Tables are rendered to Markdown honouring colspan *and* rowspan,
  kept whole through chunking, and rendered as tables again in the UI.
- **Bind both address families.** On the dev box `/etc/hosts` resolves `localhost`
  to `::1` first, while VS Code's port scanner reads the IPv4 table. A dual-stack
  IPv6 socket satisfies one and is invisible to the other.

---

## Status

Working: scraping, retrieval, three-mode answering, both guards, chat UI,
multi-turn memory, local voice both directions.

Open: streaming replies (answers take 5–18s), undergraduate and graduate fee
pages can both land in one answer, and no reboot persistence for the background
services.

GIKI publishes two live pages with contradicting admission fees (Rs. 75,000 and
Rs. 62,500). The assistant surfaces the conflict rather than silently picking one;
adjudicating the institution's own inconsistency is out of scope.
```

---
---

# Night Run 2 — 2026-07-30 (later)

Four jobs, all complete. One regression I caused, flagged below rather than
buried.

## 1. Public URL — live

```
https://weekend-determine-possibly-furthermore.trycloudflare.com
```

Verified from the box: page 200, `/capabilities` reports stt+tts, a real
question returned GROUNDED with citations, `/speak` returned audio. cloudflared
2026.7.3 installed to `~/.local/bin`, no sudo.

**The catch, and it needs a decision from you:** a *quick tunnel* mints a brand
new hostname every time cloudflared restarts — including on reboot. The URL
above is live now but will not survive a restart. `src/tunnel_url.sh` prints the
current one.

**A permanent URL needs a free Cloudflare account.** With one you get a *named*
tunnel with a fixed address that survives restarts. It requires: signing up,
`cloudflared tunnel login` (opens a browser, needs your credentials — I cannot
do this for you), then `cloudflared tunnel create giki-assistant`. A custom
domain is optional; a stable `*.cfargotunnel.com` address is not. That is the
only outstanding thing on this job.

## 2. Reboot survival — done

Four user services, linger already enabled, same pattern as `code-tunnel.service`:

| Service | stop→start | kill -9 → revived | enabled |
|---|---|---|---|
| `ollama` | active | 2923920 → 2925003 | yes |
| `giki-ui` (HTTP 8137) | active | 2924219 → 2925317 | yes |
| `giki-ui-https` (HTTPS 8443) | active | — | yes |
| `cloudflared` | active | 2924529 → 2925518 | yes |

All four `WantedBy=default.target` with `Restart=always`, so a real reboot brings
the stack up with no manual steps. The `nohup` fragility is gone.

## 3. Stale-year bug — fixed

**Before, turn 1:** answered from 2021 sources, evidence ordered `['2021',
'2021', 'undated', ...]`, and never mentioned the year.

**Before, turn 2** (`"but the sources say it's from 2021"`): restated the answer.
*"The sources used refer to the same year… All three sources indicate…"*

**Root cause of the double-down**, which was not obvious: `condense_question()`
rewrote the challenge into *"Does FES offer undergraduate programs in 2021?"* —
a content question. The system then answered content again. What looked like
stubbornness was the challenge being destroyed before it reached the model.

**After, turn 1:** evidence re-ordered `['undated', 'undated', 'undated',
'2021', '2021']` — live pages first. GROUNDED, no leak, stable over three runs.

**After, turn 2:** *"The live website pages provide information about GIKI's
current programs and admissions for 2026… The prospectus from 2021 is outdated
and does not reflect the current offerings as of 2026, but it still
establishes…"* — acknowledges, dates the sources, distinguishes live from stale.

Mechanism, following the `ended_terms()` precedent of putting the signal in the
data: `year_intent()` detects a named year or "current/latest"; `prefer_recent()`
deterministically re-orders evidence toward live and newest sources before the
model sees any of it; `provenance_line()` appends dated-source disclosure built
from metadata; `is_source_challenge()` catches the objection *before* condensing
and routes it to a prompt whose job is to acknowledge rather than re-answer.

Also removed wording that invited invention — describing undated pages as
"reflects current information" had the model claiming sources were "from July
2026".

## 4. Scaffolding leak — fixed, and the real defect was worse

The parser only recognised bare headers, so `**ANSWERED:**`, `- ANSWERED:` or a
chatty preamble line defeated it — and the empty-body fallback returned the raw
reply, which is exactly how the header reached your screen. Matching is now
tolerant of that decoration, plus an **unconditional sweep** strips any surviving
header or separator line. Verified against 10 shapes including your exact
screenshot case: **0 leaks**.

**But the leak was a symptom.** The raw reply for that question was, in full:

```
'ANSWERED: yes\nSOURCES_USED: 1, 2, 3'
```

The model emitted the header and stopped, producing no answer at all — so the
old code had nothing else to show. Removing the trailing `ANSWERED:` prime
reduced it, and an empty body now triggers one retry without the structured
contract. Three consecutive runs of that question: GROUNDED, no leak, no empty
answers.

I briefly made this worse before making it better: my first fix returned an
empty answer instead of the leak, which is worse, and the second returned REFUSE
for an answerable question. Both caught by re-testing rather than assuming.

## Regression I caused — needs your call

Validation went **44/50 → 41/50** (0 contract failures, unchanged). The
composition matters more than the number:

- **ISO quality policy** — was REFUSE, now correct. Fixed.
- **Dean** — now REFUSE where it was GROUNDED, but the answer is *better*:
  *"The sources mention Dr. Qadeer Ul Hasan as the Dean but do not specify which
  faculty he oversees."* It now finds the right person and states the exact
  ambiguity. The stored expectation is stale, not the behaviour.
- **Four questions drifted GENERAL → GROUNDED** (CV writing, job interview,
  version control, plus the pre-existing set). This is the real regression:
  removing the `ANSWERED:` prime made the model less willing to say "no", so
  borderline generic questions now get answered "grounded" on weak evidence.

The clearest example — *"Why is version control useful for software projects?"*
returns a generic answer citing `/course/software-testing-and-quality-engineering/`
and `/course/principles-of-marketing/`. The marketing citation is plainly
irrelevant. GENERAL mode would have flagged this as the model's own knowledge;
GROUNDED implies a source that is not really there.

Judged on behaviour rather than the stored expectations, roughly 46/50 — but
those three loosely-cited answers are a genuine quality loss and I would rather
you decided than have me quietly tune it at 3am. The fix is a stricter ANSWERED
criterion, which is a prompt change, and prompt changes are the thing we have
repeatedly found unreliable here.

## Left running

All four services enabled and running. Public URL live. Repo clean, pushed
through `3333dc7`. `README.md` untouched.
