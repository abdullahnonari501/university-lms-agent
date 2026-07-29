# Phase 4 Report — Stale Sources, Chat UI, Memory, Voice

Covers commits `a2c8f72..7fd9e56` (7 commits, ~940 lines added across 7 files).
Written 2026-07-30.

---

## 1. Stale sources

**The problem.** Two answers were confidently wrong in the same way: the corpus
holds superseded documents with no recency signal, and grounding alone cannot
tell current from historical.

**Built as one connected job, not two systems:**

*Foundation — index time.* Every chunk now carries `source_date` + `date_basis`,
extracted in priority order:

| Signal | Example | Result |
|---|---|---|
| Year range in filename | `Handbook-2023-24` | 2024 |
| Year in filename | `UG_Prospectus_2021` | 2021 |
| WordPress upload path | `/uploads/2023/11/` | 2023 |
| Nothing | live site page | **undated** |

Undated means *current* — the website shows what is true now, while a 2021
prospectus stays 2021 forever. Distribution: 2,944 undated, 716 from 2021,
662 from 2024, remainder smaller years.

*Behaviour — answer time.* Each source in the prompt carries a `CURRENCY:` line
(`published 2021; 5 year(s) old — may be superseded`, or `live website page,
undated — reflects current information`). The model is told to prefer newer
sources, to date any claim drawn from an older one, and never to present a former
office-holder as current. Those lines are internal, so it is also told never to
quote them back at the user.

**The part metadata could not cover.** Ahmar Rashid's page is a *live, undated*
page — the staleness is inside the text: "Dean FCSE, September 2019 to August
2023". `ended_terms()` detects role-associated date ranges that have already
passed and injects an explicit warning into the evidence itself. Three prompt
rewrites had failed to make the model notice this on its own; putting the signal
in the data worked immediately.

### Before / after

**Case 1 — Dean. FIXED.**

- Before: *"PROF. ENGR. DR. AHMAR RASHID **is the Dean** … He served from
  September 2019 to August 2023."* — a former Dean asserted as current, with
  citations, self-contradicting in one sentence.
- After: *"The sources do not name a current Dean … Source [3] mentions Dr. Ahmar
  Rashid as serving as Dean from September 2019 to August 2023, but this is an
  ended term."* + who to contact. Stable across repeat runs; **two** former
  office-holders flagged, not just one.

It still does not name Prof. Dr. Qadeer Ul Hasan, whose page says "DEAN" without
naming the faculty. Under the project's own "ambiguity resolves toward REFUSE"
rule that is correct caution, and the refusal now explains itself.

**Case 2 — the Rs. 62,500 fee. NOT A STALENESS BUG. My diagnosis was wrong.**

I had reported this figure as coming from the 2021 prospectus. The prospectus
contains "62,500" **zero times**. It comes from a live page:

| Live page | Admission fee |
|---|---|
| `/admissions/admissions-undergraduates/ugrad-fees-and-expenses/` | Rs. 75,000 |
| `/admissions/admissions-undergraduates/fees-and-expenses/` | Rs. 62,500 |

GIKI publishes two live, undated pages with contradicting fees. Dating cannot
resolve it — both are equally current. The bot now surfaces the conflict instead
of silently picking one. Malik's call: adjudicating GIKI's own inconsistency is
not this project's job.

**Residual, and this one is ours:** one run compared the undergraduate fee
against Rs. 80,000, the *graduate* admission fee. UG and graduate fee pages
reaching one answer is a retrieval-scoping bug, still open.

---

## 2. Chat UI

`src/serve.py`, **standard library only** — no Flask, no FastAPI, nothing to pip
install. The box has restricted sudo and is not permanently ours, so a server
needing no installation and no root is worth more than a nicer framework.
`uvicorn` was present but FastAPI was not, and adding a dependency to serve one
page is a poor trade.

Markdown tables render as real tables. Fee tables survive extraction, chunking
and retrieval with their row/column structure intact; flattening them to prose at
the last step would undo that work and reintroduce the exact ambiguity that made
Rs. 470,000 unreadable.

---

## 3. Conversation memory

Storing messages would not have been enough. **Retrieval has no memory** — "what
about for MS students?" embeds to nothing useful and BM25 sees only stopwords. So
`condense_question()` folds the conversation into a standalone query *before*
search; retrieval itself stays stateless, and the browser holds the transcript.

| User types | Actually searched |
|---|---|
| "What are the undergraduate fee charges?" | *(unchanged)* |
| "what about for MS students?" | "What are the fee charges for MS students?" |
| "is it refundable?" | "Is the admission fee for **MS students** refundable?" |

The third carried "MS students" across two turns.

**Issue hit:** told only to "rewrite if not standalone", the 7B model judged
*"what about for MS students?"* to already be standalone and passed it through
unchanged — 2 of 3 test follow-ups failed. Few-shot examples fixed it. Same
pattern as the `ANSWERED` header earlier in Phase 3: this model needs shown, not
told.

The server treats the posted transcript as untrusted — roles whitelisted,
content truncated, turn count capped — so a crafted payload cannot blow the
context window.

---

## 4. Voice

Both directions local: **Whisper small** (GPU, ~0.8s per utterance) for
dictation, **Piper** for replies. Nothing reaches a speech vendor, matching the
rest of the self-hosted stack.

**Why not ElevenLabs:** better voice, but per-answer cost forever, needs a card,
and ships every reply off-box. Piper gets ~80% of the quality at zero cost and
zero egress. Swapping is a one-function change in `voice.speak()` if a polished
demo video is ever wanted.

**Two design constraints solved:**

- *ffmpeg is absent and sudo is restricted.* Audio is encoded to WAV **in the
  browser** (`AudioContext` decodes, JS writes the WAV header), so Python's
  stdlib `wave` reads it with no codec dependency.
- *GPU is tight.* Qwen holds 9 of 12.3 GB. Whisper loads to GPU only when >2.5 GB
  is free, else CPU — an OOM there would take the chat down with it.

**Two bugs, both found by testing the loop rather than assuming** (Piper speaks a
phrase, Whisper transcribes it back):

1. Whisper rendered **"GIKI" as "Chiki"** — it has never seen the name. Fixed
   with a vocabulary prompt.
2. That prompt then made it hear **"fees" as "FES"**, because I had included the
   faculty acronym. Biasing toward a rare word that collides with a common one
   costs more than it gains, and "fees" is the most asked-about word in this
   corpus. Removed the clashing acronyms.

Both verified: GIKI, FCSE, CGPA and fees all transcribe correctly.

---

## 5. Access and networking — where most of the friction was

Four distinct failures, in the order they appeared:

| Symptom | Cause | Fix |
|---|---|---|
| "Checked the link from another machine, didn't work" | Bound to `127.0.0.1` — localhost only | `--host 0.0.0.0` |
| Microphone would not work at all off-box | Browsers block `getUserMedia` outside a **secure context** | Self-signed HTTPS via stdlib `ssl`, no root. Key gitignored |
| Cannot record through RustDesk | RustDesk streams the box's *screen*; the browser looks for a mic **on the box**. No server-side fix exists | Page must run on the device holding the mic |
| "Unable to forward localhost:8137" | Read like a stale registration. Actually `/etc/hosts` resolves `localhost` to **`::1` before `127.0.0.1`**, and the server was IPv4-only — invisible to anything connecting by name | Dual-stack socket: `AF_INET6` with `IPV6_V6ONLY` cleared |

The IPv6 one is the instructive failure: the error message pointed at the tunnel,
and the screenshot ruled out a stale forward (neither 8000 nor 8137 was listed).
`[::1]:8137` returning nothing while `127.0.0.1:8137` returned 200 was the actual
evidence.

Also corrected: CLAUDE.md said Cursor; the box runs a **VS Code** Remote Tunnel
named `popos`. Port-forwarding instructions differ between the two, so the wrong
note was actively misleading.

---

## Current state

| | |
|---|---|
| Corpus | 1,306 files (1,297 pages + 9 PDFs) |
| Index | 4,363 chunks, dated |
| Models | qwen2.5:7b, nomic-embed-text, whisper-small, piper |
| Endpoints | `https://…:8443` (LAN) · `http://…:8137` (for tunnel forwarding) |
| Validation | 44/50 modes, 0 contract failures |

## Open items

1. **`data/` exists only on this box** — gitignored, so corpus + index are not
   backed up. Rebuilding is a ~2-hour scrape and re-index, and CLAUDE.md notes
   the machine is not permanently ours. Highest-value remaining risk.
2. **Streaming replies** — answers take 5–18s behind a typing dot.
3. **UG/graduate fee scoping** — both fee page families can land in one answer.
4. **Nothing survives a reboot** — Ollama and the server are `nohup` processes.
