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
    for line in lines[:6]:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ANSWERED:"):
            answered = "NO" not in upper.split(":", 1)[1].strip().upper()[:3]
            header_seen = True
            consumed += 1
        elif upper.startswith("SOURCES_USED:"):
            named = [int(n) for n in re.findall(r"\d+", stripped.split(":", 1)[1])]
            header_seen = True
            consumed += 1
        elif stripped == "---" and header_seen:
            consumed += 1
            break
        elif not stripped:
            consumed += 1
        else:
            break

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

    return answered, citations, body or raw.strip()


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


def source_note(hit: dict) -> str:
    """One line telling the model how current this source is."""
    bits = []
    if hit.get("source_date"):
        bits.append(f"published {hit['source_date']}")
        if int(hit["source_date"]) < CURRENT_YEAR:
            bits.append(f"{CURRENT_YEAR - int(hit['source_date'])} year(s) old — "
                        "may be superseded")
    else:
        bits.append("live website page, undated — reflects current information")

    ended = ended_terms(hit.get("text", ""))
    if ended:
        bits.append(
            "WARNING: mentions a role with an ENDED term (" + ", ".join(ended)
            + ") — that person is a FORMER holder, not the current one"
        )
    return "; ".join(bits)


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
        "ANSWERED:"
    )


def answer(question: str, model: str = DEFAULT_MODEL, k: int = TOP_K,
           threshold: float = RELEVANCE_THRESHOLD) -> Answer:
    started = time.monotonic()
    not_found_reason = ""
    # Pull a wider pool, drop duplicate pages, then let the reranker decide the
    # order. The relevance threshold still uses the best dense score in the
    # pool, so reranking changes what is read, never how confident we are.
    pool = search(question, k=RERANK_POOL)
    top = max((h["score"] for h in pool), default=0.0)
    hits = rerank(question, dedupe_by_source(pool)[:RERANK_KEEP], model)[:k]

    if hits and top >= threshold:
        kept = fit_chunks(hits)
        raw = call_qwen(model, SYSTEM_GROUNDED, build_grounded_prompt(question, kept))
        # The prompt ends primed with "ANSWERED:", so the reply continues from
        # there rather than repeating the label. Put it back before parsing.
        if not raw.lstrip().upper().startswith("ANSWERED"):
            raw = f"ANSWERED: {raw.lstrip()}"
        # Citations resolve through chunk metadata by index, so the model can
        # only ever narrow the retrieved set -- it cannot introduce a URL.
        answered, citations, body = parse_structured(raw, kept)
        if answered:
            # Last line of defence: a name or figure that appears in no chunk is
            # fabricated, whatever the model claimed. Drop to the classifier
            # rather than emit an invented fact wearing citations.
            invented = unsupported_claims(body, kept)
            if invented:
                print(f"  [guard] dropped unsupported claim(s): {invented}",
                      file=sys.stderr, flush=True)
            else:
                return Answer("GROUNDED", body, citations, top,
                              time.monotonic() - started, len(kept), model, kept)
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

    if classify_question(model, question) == "GENERIC":
        text = call_qwen(model, SYSTEM_GENERAL, question)
        return Answer("GENERAL", f"{GENERAL_FLAG}\n\n{text}", [], top,
                      time.monotonic() - started, 0, model)

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
    return Answer("REFUSE", text, [], top, time.monotonic() - started, 0, model)


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
