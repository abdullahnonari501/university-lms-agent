"""Phase 3 Stage 2: head-to-head between the vision and text-only Qwen models.

Runs the same questions through both models on IDENTICAL retrieved chunks, so
the only variable is the model. Records mode, answer, citations, latency and
VRAM footprint.

Usage:
    python3 src/compare_models.py                 # run the battery
    python3 src/compare_models.py --probe         # just show retrieval scores
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer as ans  # noqa: E402
from retrieve import search  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "data" / "logs"
REPORT = LOGS_DIR / "phase3_model_comparison.txt"

VISION_MODEL = "qwen2.5vl:7b-q4_K_M"
TEXT_MODEL = "qwen2.5:7b"

# The 3 from Stage 1, plus 3 spanning courses, faculty and policies.
QUESTIONS = [
    ("fees (table)", "What are the undergraduate fee charges?"),
    ("generic concept", "What is a GPA and how is it calculated?"),
    ("not in corpus", "What is the hostel fee for international students?"),
    ("courses", "What are the course learning outcomes for the Data Structures course?"),
    ("faculty", "Who is the Dean of the Faculty of Computer Science and Engineering?"),
    ("policies", "What is the policy on academic dishonesty and plagiarism?"),
]


def vram_used_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


def unload(model: str) -> None:
    """Ask Ollama to drop the model so VRAM readings are not confounded."""
    try:
        body = json.dumps({"model": model, "prompt": "", "keep_alive": 0}).encode()
        req = urllib.request.Request(
            f"{ans.OLLAMA_URL}/api/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3)


def run_one(model: str, question: str, hits: list[dict]) -> dict:
    """Answer using pre-fetched hits so both models see identical evidence."""
    started = time.monotonic()
    top = hits[0]["score"] if hits else 0.0

    if hits and top >= ans.RELEVANCE_THRESHOLD:
        kept = ans.fit_chunks(hits)
        raw = ans.call_qwen(model, ans.SYSTEM_GROUNDED, ans.build_grounded_prompt(question, kept))
        if not raw.lstrip().upper().startswith("ANSWERED"):
            raw = f"ANSWERED: {raw.lstrip()}"
        answered, citations, body = ans.parse_structured(raw, kept)
        invented = ans.unsupported_claims(body, kept) if answered else []
        if answered and not invented:
            return {"mode": "GROUNDED", "text": body, "citations": citations,
                    "latency": time.monotonic() - started, "chunks": len(kept), "top": top}
        if invented:
            print(f"    [guard] {model}: dropped {invented}", flush=True)

    if ans.classify_question(model, question) == "GENERIC":
        text = ans.call_qwen(model, ans.SYSTEM_GENERAL, question)
        return {"mode": "GENERAL", "text": f"{ans.GENERAL_FLAG}\n\n{text}", "citations": [],
                "latency": time.monotonic() - started, "chunks": 0, "top": top}

    return {"mode": "REFUSE", "text": "(standard refusal text)", "citations": [],
            "latency": time.monotonic() - started, "chunks": 0, "top": top}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if args.probe:
        for label, q in QUESTIONS:
            hits = search(q, k=3)
            print(f"{hits[0]['score']:.4f}  [{label}] {q}")
            for h in hits[:2]:
                print(f"         {h['score']:.4f} [{h['category']}] {h['source_url'][:72]}")
        return 0

    # Retrieve once per question; both models then see identical evidence.
    evidence = {q: search(q, k=ans.TOP_K) for _, q in QUESTIONS}

    results: dict[str, dict[str, dict]] = {}
    vram: dict[str, int] = {}
    for model in (VISION_MODEL, TEXT_MODEL):
        unload(VISION_MODEL if model == TEXT_MODEL else TEXT_MODEL)
        print(f"\n=== {model} ===", flush=True)
        baseline = vram_used_mb()
        results[model] = {}
        peak = baseline
        for label, q in QUESTIONS:
            r = run_one(model, q, evidence[q])
            results[model][q] = r
            peak = max(peak, vram_used_mb())
            print(f"  [{r['mode']:8}] {r['latency']:5.1f}s  {label}", flush=True)
        vram[model] = peak

    L: list[str] = []
    a = L.append
    a("PHASE 3 STAGE 2 - MODEL HEAD-TO-HEAD")
    a(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    a("Identical retrieved chunks for both models; only the model differs.")
    a("=" * 78)
    a("")
    a(f"{'question':<46} {'vision mode':<10} {'text mode':<10} {'v s':>6} {'t s':>6}")
    for label, q in QUESTIONS:
        v, t = results[VISION_MODEL][q], results[TEXT_MODEL][q]
        a(f"{label + ': ' + q[:38]:<46} {v['mode']:<10} {t['mode']:<10} "
          f"{v['latency']:>6.1f} {t['latency']:>6.1f}")
    a("")
    for model in (VISION_MODEL, TEXT_MODEL):
        lat = [results[model][q]["latency"] for _, q in QUESTIONS]
        a(f"{model}: mean {sum(lat)/len(lat):.1f}s, peak VRAM {vram[model]} MiB")
    a("")
    a("=" * 78)
    a("FULL ANSWERS")
    a("=" * 78)
    for label, q in QUESTIONS:
        a("")
        a(f"### [{label}] {q}")
        for model in (VISION_MODEL, TEXT_MODEL):
            r = results[model][q]
            a("")
            a(f"--- {model}  [{r['mode']}]  {r['latency']:.1f}s  "
              f"{len(r['citations'])} citations")
            a(r["text"][:1400])
            for c in r["citations"]:
                a(f"    * {c}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
