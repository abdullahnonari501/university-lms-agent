"""Phase 3 Stage 3: validation battery over the three answer modes.

Expected modes are a hypothesis, not ground truth. --probe prints what
retrieval actually finds for each question so the expectation can be corrected
against the corpus before the battery runs -- the lesson from assuming the GPA
question would be GENERAL when the handbook covers it, and from assuming the
Dean was absent when only the thin-page filter had hidden him.

Usage:
    python3 src/validate.py --probe    # retrieval scores + expectation check
    python3 src/validate.py            # full battery -> phase3_validation.txt
"""

import argparse
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer as ans  # noqa: E402
from retrieve import search  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "data" / "logs"
OUT = LOGS_DIR / "phase3_validation.txt"

# (expected_mode, question). 28 GROUNDED, 12 GENERAL, 10 REFUSE.
QUESTIONS: list[tuple[str, str]] = [
    # ---- GROUNDED: facts the corpus should hold -------------------------
    ("GROUNDED", "What are the undergraduate fee charges?"),
    ("GROUNDED", "What is the annual tuition fee for foreign students?"),
    ("GROUNDED", "What is the admission fee and is it refundable?"),
    ("GROUNDED", "What are the graduate programme fees for MS students?"),
    ("GROUNDED", "When does the fall semester start?"),
    ("GROUNDED", "What are the admission requirements for undergraduate programs?"),
    ("GROUNDED", "What documents are needed for admission?"),
    ("GROUNDED", "What is the admission test syllabus for management sciences?"),
    ("GROUNDED", "Who is the Dean of the Faculty of Computer Science and Engineering?"),
    ("GROUNDED", "What are the qualifications of Prof. Dr. Qadeer Ul Hasan?"),
    ("GROUNDED", "Tell me about the Faculty of Computer Science and Engineering"),
    ("GROUNDED", "What undergraduate programmes does FCSE offer?"),
    ("GROUNDED", "What is the course content of Data Structures and Algorithms?"),
    ("GROUNDED", "What are the course learning outcomes listed for engineering courses?"),
    ("GROUNDED", "What credit hours does the Computational Fluid Dynamics course carry?"),
    ("GROUNDED", "What labs and facilities does the Faculty of Engineering Sciences have?"),
    ("GROUNDED", "What are the rules about student discipline?"),
    ("GROUNDED", "What is the policy on academic dishonesty and plagiarism?"),
    ("GROUNDED", "What happens if a student is caught cheating in an exam?"),
    ("GROUNDED", "What is GIKI's policy for students with disabilities?"),
    ("GROUNDED", "What is the sexual harassment policy?"),
    ("GROUNDED", "How does the GIKI transport system work?"),
    ("GROUNDED", "What scholarships are available?"),
    ("GROUNDED", "How is CGPA calculated at GIKI?"),
    ("GROUNDED", "What is the minimum CGPA to stay in good academic standing?"),
    ("GROUNDED", "How do I request an official transcript?"),
    ("GROUNDED", "What research areas does the Faculty of Computer Science work on?"),
    ("GROUNDED", "What is the ISO quality policy of the institute?"),
    # ---- GENERAL: generic, no institution-specific fact -------------------
    ("GENERAL", "How can I study more effectively for exams?"),
    ("GENERAL", "What is machine learning?"),
    ("GENERAL", "How do I write a good CV for an internship application?"),
    ("GENERAL", "What is the difference between a thesis and a project?"),
    ("GENERAL", "How should I manage my time as an engineering student?"),
    ("GENERAL", "What is the difference between a compiler and an interpreter?"),
    ("GENERAL", "How do I deal with exam stress?"),
    ("GENERAL", "What does a credit hour generally mean in a university?"),
    ("GENERAL", "How do I prepare for a technical job interview?"),
    ("GENERAL", "What are good habits for working on a group project?"),
    ("GENERAL", "Why is version control useful for software projects?"),
    ("GENERAL", "What is the general difference between an MS and a PhD?"),
    # ---- REFUSE: GIKI-specific facts the public site cannot answer --------
    ("REFUSE", "What is the WiFi password for Hostel 3?"),
    ("REFUSE", "What are my grades this semester?"),
    ("REFUSE", "What is the hostel fee for international students?"),
    ("REFUSE", "Who is the current president of the GIKI student council?"),
    ("REFUSE", "What is on the cafeteria menu today?"),
    ("REFUSE", "How many students were enrolled at GIKI in 2026?"),
    ("REFUSE", "What is my roll number?"),
    ("REFUSE", "Which room is my Thermodynamics class in tomorrow?"),
    ("REFUSE", "What was the closing merit for Computer Science in 2025?"),
    ("REFUSE", "Who won the GIKI cricket tournament last year?"),
]

# Abbreviations must not end a sentence: splitting naively put "Rs." at a
# boundary, so "The semester fee is Rs. 470,000" became a fragment with the
# figure separated from the word that gives it meaning.
SENTENCE_SPLIT_RE = re.compile(r"(?<!\bRs)(?<!\bNo)(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<=[.!?])\s+(?=[A-Z])")

CONTACT_MARKERS = ("admissions@giki.edu.pk", "contact-us", "advisor", "office")
PERIOD_MARKERS = ("per semester", "semester fee", "annual", "per year", "each semester")


def probe() -> int:
    print(f"{'exp':<9} {'top':>7}  question")
    counts: Counter[str] = Counter()
    for expected, q in QUESTIONS:
        hits = search(q, k=3)
        top = hits[0]["score"] if hits else 0.0
        predicted = "GROUNDED" if top >= ans.RELEVANCE_THRESHOLD else "(low)"
        flag = " " if (expected == "GROUNDED") == (predicted == "GROUNDED") else "!"
        counts[flag] += 1
        print(f"{flag}{expected:<8} {top:>7.4f}  {q[:62]}")
        if flag == "!":
            for h in hits[:2]:
                print(f"{'':>18}{h['score']:.4f} {h['source_url'][:66]}")
    print(f"\nmatches expectation: {counts[' ']}  mismatches: {counts['!']}")
    print("(a mismatch is not necessarily wrong -- check whether the corpus")
    print(" really covers it before changing either the question or the threshold)")
    return 0


def is_periodic_figure(figure: str, evidence: list[dict]) -> bool:
    """True when the figure sits under a per-period column in a source table.

    A number inside a Markdown row inherits its meaning from that table's
    header, so look there rather than at nearby characters. Prose figures (the
    one-time admission fee) are periodic only if their own sentence says so.
    """
    # Boundary-matched: plain containment finds "75,000" inside "175,000" and
    # attributes the wrong row's meaning to it.
    pattern = re.compile(rf"(?<![\d,]){re.escape(figure)}(?![\d,])")
    for hit in evidence:
        lines = hit["text"].splitlines()
        for i, line in enumerate(lines):
            if not pattern.search(line):
                continue
            if not line.lstrip().startswith("|"):
                # Prose: judge the sentence the figure is in. PDF extraction
                # puts many sentences on one line, so a line-wide test let
                # "Semester fee cannot be paid in installments" mark the
                # one-time admission fee in the next sentence as periodic.
                for sentence in re.split(SENTENCE_SPLIT_RE, line):
                    if pattern.search(sentence):
                        low = sentence.lower()
                        if any(m in low for m in ("semester", "annual", "per year")):
                            return True
                continue
            if line.lstrip().startswith("|"):
                for header in reversed(lines[max(0, i - 6): i]):
                    if header.lstrip().startswith("|") and any(
                        m in header.lower() for m in ("semester", "annual", "per year")
                    ):
                        return True
    return False


def check(expected: str, result: ans.Answer, question: str) -> list[str]:
    """Automatic contract checks. Returns a list of failures.

    Citations are audited against result.evidence -- the chunks the answer was
    actually built from. Comparing against a fresh search() was wrong: answer()
    retrieves a wider pool and reranks it, so a second search returns a
    different set and every legitimate citation looked like a violation.
    """
    fails = []
    retrieved = {h["source_url"] for h in result.evidence}

    if result.mode == "GROUNDED":
        if not result.citations:
            fails.append("GROUNDED with no citation")
        for c in result.citations:
            if c not in retrieved:
                fails.append(f"citation not in retrieved set: {c}")
    elif result.mode == "GENERAL":
        if ans.GENERAL_FLAG not in result.text:
            fails.append("GENERAL without the visible flag")
        if result.citations:
            fails.append("GENERAL carrying citations")
        if "giki.edu.pk" in result.text.lower():
            fails.append("GENERAL text references a GIKI URL")
    elif result.mode == "REFUSE":
        if not any(m in result.text.lower() for m in CONTACT_MARKERS):
            fails.append("REFUSE without pointing anywhere to ask")
        if result.citations:
            fails.append("REFUSE carrying citations")

    # A figure drawn from a per-period column must carry its period. Decided
    # structurally, not by proximity: the admission fee is one-time prose that
    # happens to sit next to the semester table, so a character window around
    # the number called it periodic and failed a correct answer.
    if result.mode == "GROUNDED":
        for figure in re.findall(r"\d{2,3},\d{3}", result.text):
            if is_periodic_figure(figure, result.evidence) and not any(
                m in result.text.lower() for m in PERIOD_MARKERS
            ):
                fails.append(f"periodic figure {figure} quoted without its period")
                break

    if result.mode != expected:
        fails.append(f"mode {result.mode}, expected {expected}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model", default=ans.DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=ans.RELEVANCE_THRESHOLD)
    args = ap.parse_args()

    if args.probe:
        return probe()

    L: list[str] = []
    a = L.append
    a("PHASE 3 VALIDATION BATTERY")
    a(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    a(f"Model: {args.model}   threshold: {args.threshold}")
    a("=" * 78)

    rows, wrong_mode, contract_fails = [], [], []
    for n, (expected, q) in enumerate(QUESTIONS, 1):
        result = ans.answer(q, model=args.model, threshold=args.threshold)
        fails = check(expected, result, q)
        rows.append((expected, result, q, fails))
        if any(f.startswith("mode ") for f in fails):
            wrong_mode.append((q, expected, result.mode))
        contract_fails.extend((q, f) for f in fails if not f.startswith("mode "))
        print(f"[{n:2}/{len(QUESTIONS)}] {result.mode:<8} exp {expected:<8} "
              f"{result.latency_s:5.1f}s {'FAIL' if fails else 'ok'}  {q[:46]}", flush=True)

    total = len(QUESTIONS)
    a("")
    a(f"Mode accuracy ......... {total - len(wrong_mode)}/{total}")
    a(f"Contract failures ..... {len(contract_fails)}")
    lat = [r[1].latency_s for r in rows]
    a(f"Latency ............... mean {sum(lat)/len(lat):.1f}s, max {max(lat):.1f}s")
    a("")
    if wrong_mode:
        a("MODE MISMATCHES")
        for q, exp, got in wrong_mode:
            a(f"  expected {exp:<9} got {got:<9} {q}")
        a("")
    if contract_fails:
        a("CONTRACT FAILURES")
        for q, f in contract_fails:
            a(f"  {f}  <- {q}")
        a("")
    a("=" * 78)
    a("FULL LOG")
    a("=" * 78)
    for expected, result, q, fails in rows:
        a("")
        a(f"Q: {q}")
        a(f"   expected={expected} got={result.mode} score={result.top_score:.4f} "
          f"chunks={result.chunks_used} latency={result.latency_s:.1f}s")
        if fails:
            a(f"   FAILURES: {fails}")
        a(f"   {result.text[:900]}")
        for c in result.citations:
            a(f"     * {c}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nmode accuracy {total - len(wrong_mode)}/{total}, "
          f"{len(contract_fails)} contract failures -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
