"""Phase 3: question in -> answer out, grounded in the GIKI corpus.

Three modes, always reported to the caller:

  GROUNDED  retrieval found relevant chunks; the answer comes only from them
            and carries citations copied verbatim from chunk metadata.
  GENERAL   retrieval found nothing relevant and the question is generic;
            answered from the model's own knowledge behind a visible flag,
            never with a GIKI citation.
  REFUSE    retrieval found nothing relevant and the question asks for a
            GIKI-specific fact. Say so and point at who to ask.

A plausible invented fee or deadline is the worst thing this bot can emit, so
ambiguity always resolves toward REFUSE.

Usage:
    python3 src/answer.py "what are the undergraduate fee charges?"
    python3 src/answer.py --model qwen2.5:7b "..."
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import search  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b"  # Malik's pick: faster, lighter, more concise than the VL model

# Measured on this box, not taken from the model card. `ollama show` reports a
# 128000 context; the server actually allocated n_ctx_slot = 4096 by default.
# 8192 is verified working and leaves VRAM headroom for a second model to load
# (16384 also worked but reached 10.8 of 12.3 GB).
NUM_CTX = 8192
RESPONSE_RESERVE = 800
CHUNK_BUDGET = 6000

TOP_K = 5
# A similarity floor, not a decision rule. Measured over the 50-question
# battery, the three modes' score distributions overlap almost entirely:
#   GROUNDED  min 0.666  median 0.786  max 0.850
#   GENERAL   min 0.634  median 0.699  max 0.846
#   REFUSE    min 0.582  median 0.690  max 0.774
# The lowest GROUNDED question scores below the highest REFUSE one, so no cutoff
# can separate them and tuning the number is wasted effort. Whether the evidence
# answers the question is decided by reading the evidence -- the model's
# ANSWERED signal plus unsupported_claims() -- not by a proxy score. This floor
# only skips a grounded attempt when retrieval returned nothing resembling the
# question at all.
RELEVANCE_THRESHOLD = 0.50

GENERAL_FLAG = "⚠ Not from GIKI's website — general knowledge:"

SYSTEM_GROUNDED = """You are a helpful assistant for students of GIK Institute (GIKI), Pakistan.

Answer the question using ONLY the numbered sources provided.

Every source carries a CURRENCY line. Obey it:
- A source marked "may be superseded" is older. If a newer source or a live
  undated page gives a different figure for the same thing, use the newer one.
- If your answer relies on a dated source, say so in the answer: "As of the 2021
  prospectus, ...". Never present an old figure as if it were current.
- If a source carries a WARNING about an ENDED term, that person is a FORMER
  holder. Never write "X is the Dean" about them. Say "X served as Dean from
  2019 to 2023". If no source names a current holder, say the sources do not
  name the current one rather than offering a predecessor as the answer.

If the student pushes back that a source is old or from a particular year,
acknowledge it directly -- say which year the sources are from and what they do
and do not show for the year asked. Never repeat the previous answer as if the
objection had not been raised.

If the question names a year and the sources predate it, say so plainly: the
sources describe an earlier year and may not reflect that one.

The CURRENCY lines are internal notes for you. Never mention them, never write
"according to the WARNING" or "the CURRENCY line says" -- state the fact itself
("Dr. X served as Dean from 2019 to 2023").

Other rules:
- Use only facts present in the sources. Never add facts from your own knowledge.
- When you quote any number from a table, state its qualifiers from the row
  label and column heading: the unit, the period, and the category. Write
  "Rs. 470,000 per semester for Engineering & Computing", never a bare number.
- Be complete: if the question asks about charges, dates or requirements in
  general, give every relevant item the sources list, not just the first one.
  No preamble and no padding, but do not omit facts the question asked for.
- Refer to sources inline as [1], [2] where relevant.
- Never invent a URL.
- Never state a person's name unless that exact name appears in the sources."""

SYSTEM_GENERAL = """You are a helpful assistant. Answer the question from your own general
knowledge, concisely and accurately.

This question is NOT about GIK Institute specifically. Do not mention GIKI, do
not cite any GIKI page, and do not claim your answer reflects GIKI policy."""

CLASSIFY_PROMPT = """Decide whether the question asks for a fact specific to one particular
university/institution (its fees, dates, staff, policies, courses, facilities,
deadlines, procedures), or whether it is a generic question that any teacher or
reference book could answer.

Answer with exactly one word: SPECIFIC or GENERIC.

Question: {question}
Answer:"""


@dataclass
class Answer:
    mode: str
    text: str
    citations: list[str] = field(default_factory=list)
    top_score: float = 0.0
    latency_s: float = 0.0
    chunks_used: int = 0
    model: str = DEFAULT_MODEL
    # The chunks actually placed in context. Callers auditing citations must
    # check against these, not against a fresh search() -- reranking means a
    # second search returns a different set.
    evidence: list[dict] = field(default_factory=list)
    # What retrieval actually searched for. Differs from the user's words on a
    # follow-up, and showing it is how a user understands a surprising answer.
    search_query: str = ""

    def render(self) -> str:
        out = [f"[{self.mode}]  (top score {self.top_score:.4f}, "
               f"{self.chunks_used} chunks, {self.latency_s:.1f}s, {self.model})", ""]
        out.append(self.text)
        if self.citations:
            out.append("")
            out.append("Sources:")
            out.extend(f"  [{i}] {c}" for i, c in enumerate(self.citations, 1))
        return "\n".join(out)


def approx_tokens(text: str) -> int:
    """Same conservative estimate the indexer uses: dense technical text and
    table rows tokenize far denser than words*1.33 alone suggests."""
    return max(int(len(text.split()) * 1.33), int(len(text) / 3.0)) + 1


def call_qwen(model: str, system: str, prompt: str, timeout: int = 600) -> str:
    body = json.dumps({
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()).get("response", "").strip()


def classify_question(model: str, question: str) -> str:
    """SPECIFIC (about one institution) or GENERIC. Anything unclear -> SPECIFIC,
    because SPECIFIC routes to REFUSE, which is the safe direction."""
    try:
        raw = call_qwen(model, "", CLASSIFY_PROMPT.format(question=question), timeout=120)
    except Exception:  # noqa: BLE001 - a classifier outage must not answer unsafely
        return "SPECIFIC"
    token = raw.strip().upper()
    if token.startswith("GENERIC"):
        return "GENERIC"
    return "SPECIFIC"


def fit_chunks(hits: list[dict]) -> list[dict]:
    """Take chunks by descending score until the context budget is spent.

    A single table chunk can run to ~1800 tokens, so five of them would blow an
    8192 slot on their own. Trimming by score keeps the best evidence.
    """
    kept, used = [], 0
    for hit in hits:
        cost = approx_tokens(hit["text"]) + 40  # + source line and numbering
        if kept and used + cost > CHUNK_BUDGET:
            break
        kept.append(hit)
        used += cost
    return kept


# Tolerant of the decorations the model puts around the header: "**ANSWERED:**",
# "- ANSWERED:", "# ANSWERED:", leading whitespace.
HEADER_LINE_RE = re.compile(
    r"^[\s\-*#>`]*\**\s*(ANSWERED|SOURCES_USED)\s*:?\**\s*(.*?)\s*$",
    re.IGNORECASE,
)
SEPARATOR_RE = re.compile(r"^[\s*]*-{2,}[\s*]*$")


def strip_scaffolding(text: str) -> str:
    """Remove any surviving header/separator lines, wherever they ended up.

    The guarantee, not the optimisation. Parsing recognises the common shapes,
    but a model can decorate the header in ways no matcher anticipates -- bold,
    bulleted, or preceded by a chatty line -- and every one of those dumped the
    raw scaffolding into the user's answer bubble. Sweeping the final body means
    a parse miss degrades to a slightly odd answer, never to leaked machinery.
    """
    kept = []
    for line in text.splitlines():
        if HEADER_LINE_RE.match(line):
            continue
        if SEPARATOR_RE.match(line) and not kept:
            continue          # separator before any real content
        kept.append(line)
    return "\n".join(kept).strip()


def parse_structured(raw: str, hits: list[dict]) -> tuple[bool, list[str], str]:
    """Pull the ANSWERED / SOURCES_USED header off the reply.

    A structured field is used rather than string-matching English phrasing,
    which varies between models and prompt tweaks and would rot silently.
    Returns (answered, citations, body).

    On an unparseable header we assume answered=True and fall back to citing
    every retrieved source: that preserves the previous, safe behaviour rather
    than discarding a real answer.
    """
    answered = True
    named: list[int] = []
    body = raw
    header_seen = False

    lines = raw.splitlines()
    consumed = 0
    # Scan a few lines rather than stopping at the first non-header: the model
    # sometimes writes a sentence before the header it was asked to lead with.
    for line in lines[:8]:
        stripped = line.strip()
        match = HEADER_LINE_RE.match(line)
        if match:
            field, value = match.group(1).upper(), match.group(2)
            if field == "ANSWERED":
                answered = "NO" not in value.strip().upper()[:3]
            else:
                named = [int(n) for n in re.findall(r"\d+", value)]
            header_seen = True
            consumed += 1
        elif SEPARATOR_RE.match(line) and header_seen:
            consumed += 1
            break
        elif not stripped:
            consumed += 1
        elif header_seen:
            break
        else:
            consumed += 1   # chatty preamble before the header; drop it

    if header_seen:
        rest = lines[consumed:]
        # The model sometimes repeats the header after the --- separator. Strip
        # any stray header lines rather than leaking them into the answer.
        while rest and (
            rest[0].strip().upper().startswith(("ANSWERED:", "SOURCES_USED:"))
            or rest[0].strip() in {"---", ""}
        ):
            rest.pop(0)
        body = "\n".join(rest).strip()

    # Unconditional final sweep -- see strip_scaffolding(). Applied outside the
    # header branch so the no-header path is covered too.
    body = strip_scaffolding(body)

    # Validate: a cited URL must be one we actually retrieved. Anything the
    # model names outside that set is dropped, so it can never introduce a URL.
    retrieved = [h["source_url"] for h in hits]
    citations: list[str] = []
    for n in named:
        if 1 <= n <= len(retrieved):
            url = retrieved[n - 1]
            if url and url not in citations:
                citations.append(url)

    if not citations and answered:
        for url in retrieved:  # safe fallback: disclose everything we searched
            if url and url not in citations:
                citations.append(url)

    # The empty-body fallback must be swept as well: returning raw here is
    # precisely how the scaffolding reached the user's screen.
    return answered, citations, body or strip_scaffolding(raw)


NAME_PATTERN = re.compile(r"\b(?:Dr|Prof|Professor|Mr|Ms|Mrs|Engr)\.?\s+([A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){0,3})")
# Real figures only: thousands-separated (470,000) or 4+ bare digits. Without
# the group-of-three requirement this matched "1,2,3,4" out of citation markers
# like "[1], [2], [3] and [4]" and rejected perfectly good answers.
NUMBER_PATTERN = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d{4,})(?![\w])")
# Titles and degrees are not part of a name. "PhD" appears somewhere in almost
# every chunk, so counting it as a name part made any invented name look
# supported -- which is exactly how "Dr. Wahid Haqim PhD" slipped through.
NAME_NOISE = {"phd", "prof", "professor", "dr", "mr", "ms", "mrs", "engr",
              "the", "and", "from", "of", "at", "is"}


def unsupported_claims(body: str, hits: list[dict]) -> list[str]:
    """Names and large figures in the answer that appear in no retrieved chunk.

    Both 7B models invented a Dean's name for a page that reads "MESSAGE FROM
    THE DEAN" but names nobody -- and returned it as GROUNDED with citations,
    which is the most dangerous shape a wrong answer can take. Prompting alone
    did not prevent it, so verify the claim against the evidence directly.

    Deliberately narrow: only titled person-names and 4+ digit numbers, the two
    things this bot must never invent. Prose is left to the prompt rules.
    """
    # Must cover everything the model was shown, not just chunk bodies: the
    # prompt includes each source's title and URL, so a year or name taken from
    # a title ("Student Handbook 2023-24") is properly grounded, and checking
    # text alone rejected correct answers.
    haystack = " ".join(
        f"{h.get('title', '')} {h.get('source_url', '')} {h['text']}" for h in hits
    )
    normalised = re.sub(r"\s+", " ", haystack).lower()
    # Separate haystack for figures with separators stripped, so "470,000" in
    # the answer matches "470,000" or "470000" in the source. Collapsing commas
    # to spaces instead turned the source into "470 000" and rejected the
    # correct fee answer.
    numeric = re.sub(r"[,\s]", "", haystack)
    bad: list[str] = []

    for name in NAME_PATTERN.findall(body):
        parts = [p for p in re.split(r"\s+", name)
                 if len(p) > 2 and p.lower().strip(".") not in NAME_NOISE]
        if not parts:
            continue
        # Require the surname itself to appear, on a word boundary. Plain
        # substring matching let "Dr. Umar Haq" through because "haq" sits
        # inside "Ghulam Ishaq Khan Institute", which appears in nearly every
        # chunk -- so any name ending in Haq looked supported.
        surname = re.escape(parts[-1].lower().strip(".,"))
        if not re.search(rf"\b{surname}\b", normalised):
            bad.append(name)

    for number in NUMBER_PATTERN.findall(body):
        if number.replace(",", "") not in numeric:
            bad.append(number)

    return bad


RERANK_POOL = 20        # candidates pulled before reranking
RERANK_KEEP = 12        # candidates actually shown to the reranker

RERANK_PROMPT = """Rank the numbered passages by how directly they answer the question.

Question: {question}

{passages}

Reply with only the passage numbers, best first, comma separated. Include just
the ones that genuinely help; omit the rest. Example: 3, 1, 7

Answer:"""


MAX_PER_SOURCE = 3


def dedupe_by_source(hits: list[dict], max_per_source: int = MAX_PER_SOURCE) -> list[dict]:
    """Cap how many chunks one source page may contribute.

    One chunk per page was too strict: the Student Handbook is a 103 KB document
    of ~100 chunks, so keeping only its best-scoring chunk usually dropped the
    section that actually answered the question -- discipline and transport both
    came back as "the sources do not contain this" while the handbook and the
    transport policy sat in the corpus. Capping instead of deduping keeps a long
    document from monopolising the top-k (one profile held ranks 2, 5 and 7 on
    the Dean query) while still letting it contribute several sections.
    """
    counts: dict[str, int] = {}
    out = []
    for hit in hits:
        key = hit.get("source_url", "")
        if counts.get(key, 0) >= max_per_source:
            continue
        counts[key] = counts.get(key, 0) + 1
        out.append(hit)
    return out


def rerank(question: str, hits: list[dict], model: str) -> list[dict]:
    """Reorder candidates by asking the model which actually answer the question.

    Similarity ranks passages by topical closeness, which is not the same as
    containing the answer: for "Who is the Dean of FCSE?" every FCSE page and
    every staff profile is topically close, while exactly one says "DEAN" next
    to a name. A reranker reads the candidates and judges that directly.

    Falls back to the original order on any failure -- reranking is an
    improvement, never a dependency.
    """
    if len(hits) < 2:
        return hits
    passages = "\n\n".join(
        f"[{i}] {' '.join(h['text'].split())[:400]}" for i, h in enumerate(hits, 1)
    )
    try:
        raw = call_qwen(model, "", RERANK_PROMPT.format(question=question, passages=passages),
                        timeout=180)
    except Exception:  # noqa: BLE001
        return hits

    order = [int(n) for n in re.findall(r"\d+", raw) if 1 <= int(n) <= len(hits)]
    if not order:
        return hits
    seen: set[int] = set()
    ranked = []
    for n in order:
        if n not in seen:
            seen.add(n)
            ranked.append(hits[n - 1])
    # Anything the reranker omitted keeps its original relative order at the back.
    ranked.extend(h for i, h in enumerate(hits, 1) if i not in seen)
    return ranked


CURRENT_YEAR = datetime.now(timezone.utc).year
# "Dean, 2019-2023", "served 2019 to 2023", "(September 2019 to August 2023)"
# A month may sit between the years -- the real case reads "September 2019 to
# August 2023", which a bare "YYYY to YYYY" pattern silently missed.
PAST_TERM_RE = re.compile(
    r"((?:19|20)\d{2})\s*(?:[-\u2013]|to|until|through)\s*"
    r"(?:[A-Za-z]+\s+)?((?:19|20)\d{2})",
    re.IGNORECASE,
)
ROLE_WORDS = ("dean", "head", "chair", "director", "rector", "president",
              "coordinator", "in charge", "incharge")


def ended_terms(text: str) -> list[str]:
    """Date ranges in the text that describe a role and have already ended.

    This is the Ahmar Rashid case: his profile is a live, undated page, so
    source-date metadata cannot help -- the staleness is *inside* the text
    ("Dean FCSE, September 2019 to August 2023"). Surfacing it as an explicit
    note beats asking the model to notice it, which three prompt rewrites
    failed to achieve.
    """
    found = []
    for match in PAST_TERM_RE.finditer(text):
        end_year = int(match.group(2))
        if end_year >= CURRENT_YEAR:
            continue
        window = text[max(0, match.start() - 120): match.end() + 40].lower()
        if any(word in window for word in ROLE_WORDS):
            found.append(f"{match.group(1)}-{match.group(2)}")
    return found


YEAR_IN_QUESTION_RE = re.compile(r"\b(20\d{2})\b")
CURRENCY_WORDS_RE = re.compile(
    r"\b(current|currently|latest|now|nowadays|this year|these days|up to date|"
    r"up-to-date|present|today)\b", re.IGNORECASE)


def year_intent(question: str) -> tuple[int | None, bool]:
    """(year named in the question, whether it asks for current state).

    Both make a question date-sensitive: "in 2026" and "what are the current
    fees" must not be answered from a 2021 prospectus without saying so.
    """
    match = YEAR_IN_QUESTION_RE.search(question)
    return (int(match.group(1)) if match else None,
            bool(CURRENCY_WORDS_RE.search(question)))


def prefer_recent(hits: list[dict]) -> list[dict]:
    """Stable re-sort putting current sources first.

    Undated means a live site page, which reflects what is true now, so it
    outranks any dated document; dated sources then order newest first. Applied
    only to date-sensitive questions, and done here rather than asked of the
    model, because the model cannot be relied on to weigh recency itself.
    """
    def key(item: tuple[int, dict]) -> tuple[int, int, int]:
        i, hit = item
        raw = hit.get("source_date") or ""
        if not raw:
            return (0, 0, i)                 # undated == current, goes first
        try:
            return (1, -int(raw), i)         # then newest dated
        except ValueError:
            return (1, 0, i)
    return [h for _, h in sorted(enumerate(hits), key=key)]


def provenance_line(hits: list[dict], body: str) -> str:
    """A dated-source disclosure built from metadata, not from model compliance.

    The model was told to date its claims and did not: asked about 2026 it
    answered from the 2021 prospectus and said nothing about the year. Stating
    it deterministically is the only version that cannot be ignored.
    """
    years: dict[str, None] = {}
    for hit in hits:
        raw = hit.get("source_date") or ""
        if raw and raw not in years:
            years[raw] = None
    if not years:
        return ""
    # Already dated in prose? Then adding a second note is noise.
    if all(y in body for y in years):
        return ""
    listed = ", ".join(
        f"{y} ({CURRENT_YEAR - int(y)} yr old)" if int(y) < CURRENT_YEAR else y
        for y in sorted(years, reverse=True)
    )
    return f"\n\nSource dates: {listed}. Newer information may supersede this."


def source_note(hit: dict) -> str:
    """One line telling the model how current this source is."""
    bits = []
    if hit.get("source_date"):
        bits.append(f"published {hit['source_date']}")
        if int(hit["source_date"]) < CURRENT_YEAR:
            bits.append(f"{CURRENT_YEAR - int(hit['source_date'])} year(s) old — "
                        "may be superseded")
    else:
        # Deliberately gives no date: saying "current" invited the model to
        # invent one ("from July 2026") for pages that carry no date at all.
        bits.append("live website page, no publication date, kept up to date")

    ended = ended_terms(hit.get("text", ""))
    if ended:
        bits.append(
            "WARNING: mentions a role with an ENDED term (" + ", ".join(ended)
            + ") — that person is a FORMER holder, not the current one"
        )
    return "; ".join(bits)


MAX_HISTORY_TURNS = 6      # keep the last few exchanges; older context rarely helps
HISTORY_CHAR_CAP = 1200    # and never let it crowd out the retrieved evidence

CONDENSE_PROMPT = """Rewrite the follow-up as a question that can be understood with no
conversation history, by copying the missing subject in from the conversation.

A follow-up is NOT standalone if it starts with "what about", "and", "how
about", or contains "it", "that", "them", "those", "there" referring to
something said earlier. Those must be rewritten.

Examples:
  Earlier: "What are the undergraduate fee charges?"
  Follow-up: "what about for MS students?"
  Standalone question: What are the fee charges for MS students?

  Earlier: "What is the admission fee?"
  Follow-up: "is it refundable?"
  Standalone question: Is the admission fee refundable?

  Earlier: "Tell me about FCSE."
  Follow-up: "who is the dean there?"
  Standalone question: Who is the dean of FCSE?

  Earlier: "What are the hostel rules?"
  Follow-up: "What scholarships are available?"
  Standalone question: What scholarships are available?

Output only the question.

Conversation so far:
{history}

Follow-up: {question}

Standalone question:"""


def format_history(history: list[dict], cap: int = HISTORY_CHAR_CAP) -> str:
    """Recent turns as plain text, newest kept, oldest dropped under the cap."""
    lines = []
    for turn in history[-MAX_HISTORY_TURNS * 2:]:
        role = "Student" if turn.get("role") == "user" else "Assistant"
        text = " ".join(str(turn.get("content", "")).split())
        lines.append(f"{role}: {text[:400]}")
    out = "\n".join(lines)
    return out[-cap:] if len(out) > cap else out


SOURCE_CHALLENGE_RE = re.compile(
    r"(that'?s|thats|it'?s|its|they'?re|sources?|prospectus|document|page|info\w*)"
    r".{0,40}\b(from\s+20\d{2}|20\d{2}|old|outdated|out of date|stale|"
    r"not current|no longer)\b"
    r"|^\s*(but|however|isn'?t|aren'?t|wasn'?t)\b.{0,60}\b(20\d{2}|old|outdated)\b",
    re.IGNORECASE,
)


def is_source_challenge(question: str) -> bool:
    """True when the student is disputing the currency of the last answer.

    This must be caught *before* condensing. "but the sources say it's from
    2021" gets rewritten into "Does FES offer undergraduate programs in 2021?"
    -- a content question -- so the system answered content again, which is
    exactly what a double-down looks like from the outside.
    """
    return bool(SOURCE_CHALLENGE_RE.search(question.strip()))


SYSTEM_CHALLENGE = """You are a helpful assistant for students of GIK Institute (GIKI), Pakistan.

The student is questioning how current your previous answer was. Do NOT simply
repeat that answer.

- Acknowledge the objection in your first sentence.
- State plainly which years the sources below are from, using their CURRENCY
  lines. Sources with no publication date are live website pages -- describe
  them as "the live website", never as being from a particular year.
- Say what those sources do and do not establish for the year the student asked
  about. If nothing covers that year, say so.
- Be brief and direct. Do not be defensive.

Never mention the CURRENCY lines themselves -- state the years as fact."""


def condense_question(question: str, history: list[dict], model: str) -> str:
    """Turn a follow-up into something retrievable on its own.

    Retrieval has no memory: "what about for MS students?" embeds to nothing
    useful and BM25 finds only the stopwords. The conversation has to be folded
    into the query *before* search, not after -- this is the piece that makes
    multi-turn work at all, rather than just displaying old messages.
    """
    if not history:
        return question
    try:
        rewritten = call_qwen(
            model, "",
            CONDENSE_PROMPT.format(history=format_history(history), question=question),
            timeout=120,
        ).strip().strip('"')
    except Exception:  # noqa: BLE001 - a condense failure must not lose the turn
        return question
    # A rewrite that collapses or rambles is worse than the original.
    if not rewritten or len(rewritten) > 300:
        return question
    return rewritten.splitlines()[0].strip()


def build_grounded_prompt(question: str, hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(
            f"[{i}] {hit['title']}\nURL: {hit['source_url']}\n"
            f"CURRENCY: {source_note(hit)}\n{hit['text']}"
        )
    # The output-format contract is repeated here, after the sources, because
    # this 7B model dropped it entirely when it lived only in the system prompt.
    return (
        "SOURCES:\n\n"
        + "\n\n---\n\n".join(parts)
        + f"\n\nQUESTION: {question}\n\n"
        "Reply in exactly this format, starting on the first line:\n\n"
        "ANSWERED: <yes if the sources above contain facts answering the "
        "question, otherwise no>\n"
        "SOURCES_USED: <comma-separated numbers of the sources you drew facts "
        "from, empty if none>\n"
        "---\n"
        "<your answer>\n\n"
        "If ANSWERED is no, say briefly what the sources are missing.\n\n"
        "The answer itself must come after the --- line. Do not stop at the "
        "header.\n"
    )


def answer(question: str, model: str = DEFAULT_MODEL, k: int = TOP_K,
           threshold: float = RELEVANCE_THRESHOLD,
           history: list[dict] | None = None) -> Answer:
    started = time.monotonic()
    not_found_reason = ""
    # Follow-ups are resolved against the conversation before anything is
    # searched. Retrieval itself is stateless.
    challenge = bool(history) and is_source_challenge(question)
    if challenge:
        # Keep the student's words, but retrieve against what they were actually
        # asking about, so the same evidence is on the table to be dated.
        prior = [h["content"] for h in (history or []) if h.get("role") == "user"]
        search_query = prior[-1] if prior else question
    else:
        search_query = condense_question(question, history or [], model)

    # Pull a wider pool, drop duplicate pages, then let the reranker decide the
    # order. The relevance threshold still uses the best dense score in the
    # pool, so reranking changes what is read, never how confident we are.
    pool = search(search_query, k=RERANK_POOL)
    top = max((h["score"] for h in pool), default=0.0)
    hits = rerank(search_query, dedupe_by_source(pool)[:RERANK_KEEP], model)[:k]

    # Date-sensitive questions get their evidence re-ordered toward current
    # sources before the model sees any of it.
    named_year, wants_current = year_intent(f"{question} {search_query}")
    if named_year or wants_current:
        hits = prefer_recent(hits)

    if hits and top >= threshold:
        kept = fit_chunks(hits)
        system = SYSTEM_CHALLENGE if challenge else SYSTEM_GROUNDED
        asked = question if challenge else search_query
        raw = call_qwen(model, system, build_grounded_prompt(asked, kept))
        # Citations resolve through chunk metadata by index, so the model can
        # only ever narrow the retrieved set -- it cannot introduce a URL.
        answered, citations, body = parse_structured(raw, kept)

        # The model sometimes emits the header and stops, producing no answer at
        # all. That is the real defect behind the scaffolding appearing on
        # screen: with nothing after the header, the old code displayed the
        # header. Retry once without the structured contract rather than
        # refusing a question the sources can answer.
        if answered and not body.strip():
            retry = call_qwen(model, system, (
                build_grounded_prompt(asked, kept).split("Reply in exactly this format")[0]
                + "Answer the question directly. Do not include any header or "
                  "metadata lines."))
            body = strip_scaffolding(retry)
            citations = [h["source_url"] for h in kept if h.get("source_url")]
            print("  [retry] model returned header only; re-asked without the "
                  "structured contract", file=sys.stderr, flush=True)
        if not body.strip():
            answered = False
        if answered:
            # Last line of defence: a name or figure that appears in no chunk is
            # fabricated, whatever the model claimed. Drop to the classifier
            # rather than emit an invented fact wearing citations.
            invented = unsupported_claims(body, kept)
            if invented:
                print(f"  [guard] dropped unsupported claim(s): {invented}",
                      file=sys.stderr, flush=True)
            else:
                # Disclose dated sources from metadata. The prompt asks for this
                # too, but only the deterministic version actually happens.
                cited = [h for h in kept if h.get("source_url") in set(citations)] or kept
                body += provenance_line(kept if challenge else cited, body)
                return Answer("GROUNDED", body, citations, top,
                              time.monotonic() - started, len(kept), model, kept,
                              search_query)
        # Retrieval was topically close but held no answer. Treat that exactly
        # like a below-threshold miss and let the classifier decide, so
        # "how do I improve my CGPA" can still fall to GENERAL rather than
        # dead-ending in a refusal. Keep the model's explanation: when it
        # declines because a source shows only a FORMER office-holder, saying
        # so is far more use to a student than boilerplate.
        # The explanation is model-generated prose and must clear the same
        # fabrication bar as an answer -- passing it through unchecked let an
        # invented name ("Ehtisham ul Haq") reach the user inside a refusal,
        # which is the one place nobody would think to look for one.
        not_found_reason = "" if unsupported_claims(body, kept) else body

    if classify_question(model, search_query) == "GENERIC":
        text = call_qwen(model, SYSTEM_GENERAL, search_query)
        return Answer("GENERAL", f"{GENERAL_FLAG}\n\n{text}", [], top,
                      time.monotonic() - started, 0, model, [], search_query)

    preamble = (
        f"{not_found_reason}\n\n" if not_found_reason else
        "I couldn't find this on GIK Institute's public website, so I don't want "
        "to guess — an invented fee, date or policy would be worse than no answer.\n\n"
    )
    text = (
        preamble +
        "For the current answer, check with the source that owns this information:\n"
        "  • Admissions Office — admissions@giki.edu.pk\n"
        "  • Student Affairs (hostel, transport, clearance) — via giki.edu.pk/contact-us/\n"
        "  • Your academic advisor or the relevant faculty office"
    )
    return Answer("REFUSE", text, [], top, time.monotonic() - started, 0, model,
                  [], search_query)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="*")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--threshold", type=float, default=RELEVANCE_THRESHOLD)
    args = ap.parse_args()

    if not args.question:
        ap.print_help()
        return 1
    print(answer(" ".join(args.question), args.model, args.k, args.threshold).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
