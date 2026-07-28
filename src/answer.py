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
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import search  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5vl:7b-q4_K_M"

# Measured on this box, not taken from the model card. `ollama show` reports a
# 128000 context; the server actually allocated n_ctx_slot = 4096 by default.
# 8192 is verified working and leaves VRAM headroom for a second model to load
# (16384 also worked but reached 10.8 of 12.3 GB).
NUM_CTX = 8192
RESPONSE_RESERVE = 800
CHUNK_BUDGET = 6000

TOP_K = 5
# Tuned empirically in Stage 3; see phase3_run_summary.txt. Strong topical hits
# in this corpus score ~0.70-0.79, clear misses ~0.60-0.66.
RELEVANCE_THRESHOLD = 0.68

GENERAL_FLAG = "⚠ Not from GIKI's website — general knowledge:"

SYSTEM_GROUNDED = """You are a helpful assistant for students of GIK Institute (GIKI), Pakistan.

Answer the question using ONLY the numbered sources below. Rules:
- Use only facts present in the sources. Never add facts from your own knowledge.
- If the sources contain a table, read it carefully: match each number to its
  column heading and row label before stating it.
- If the sources do not fully answer the question, say exactly what they do and
  do not cover.
- Be concise and direct. No preamble.
- Refer to sources inline as [1], [2] where relevant.
- Never invent a URL."""

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


def build_grounded_prompt(question: str, hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[{i}] {hit['title']}\nURL: {hit['source_url']}\n{hit['text']}")
    return "SOURCES:\n\n" + "\n\n---\n\n".join(parts) + f"\n\nQUESTION: {question}\n\nANSWER:"


def answer(question: str, model: str = DEFAULT_MODEL, k: int = TOP_K,
           threshold: float = RELEVANCE_THRESHOLD) -> Answer:
    started = time.monotonic()
    hits = search(question, k=k)
    top = hits[0]["score"] if hits else 0.0

    if hits and top >= threshold:
        kept = fit_chunks(hits)
        text = call_qwen(model, SYSTEM_GROUNDED, build_grounded_prompt(question, kept))
        # Citations are built from chunk metadata, never parsed out of the
        # model's text -- that makes a fabricated or altered URL impossible.
        citations: list[str] = []
        for hit in kept:
            if hit["source_url"] and hit["source_url"] not in citations:
                citations.append(hit["source_url"])
        return Answer("GROUNDED", text, citations, top, time.monotonic() - started,
                      len(kept), model)

    if classify_question(model, question) == "GENERIC":
        text = call_qwen(model, SYSTEM_GENERAL, question)
        return Answer("GENERAL", f"{GENERAL_FLAG}\n\n{text}", [], top,
                      time.monotonic() - started, 0, model)

    text = (
        "I couldn't find this on GIK Institute's public website, so I don't want "
        "to guess — an invented fee, date or policy would be worse than no answer.\n\n"
        "Please check with the source that owns this information:\n"
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
