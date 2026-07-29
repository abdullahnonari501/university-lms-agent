"""Phase 4: a chat UI over answer.answer().

Standard library only -- no Flask, no FastAPI, nothing to pip install. The box
runs a restricted sudo and is not permanently ours, so a server that needs no
installation and no root is worth more than a nicer framework.

Single-turn by design: Phase 3 fixed answering at one question at a time, and
conversation memory is a separate piece of work.

    python3 src/serve.py            # http://127.0.0.1:8000
    python3 src/serve.py --port 8100 --host 0.0.0.0
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer as ans  # noqa: E402

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GIKI Assistant</title>
<style>
  :root {
    --bg:#faf9f7; --fg:#1c1b19; --muted:#6b6862; --line:#e3e0da;
    --card:#fff; --accent:#1c5d3a; --warn:#8a5a00; --stop:#6b2d2d;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16151a; --fg:#eceae6; --muted:#9b968e; --line:#2e2c33;
            --card:#1e1d23; --accent:#7bc49a; --warn:#e0b45f; --stop:#e08c8c; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:760px; margin:0 auto; padding:32px 20px 80px; }
  h1 { font-size:20px; margin:0 0 4px; letter-spacing:-.01em; }
  .sub { color:var(--muted); font-size:14px; margin:0 0 28px; }
  form { display:flex; gap:8px; margin-bottom:8px; }
  input[type=text] { flex:1; padding:12px 14px; font-size:16px; color:var(--fg);
    background:var(--card); border:1px solid var(--line); border-radius:10px; }
  input[type=text]:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  button { padding:12px 18px; font-size:15px; font-weight:600; cursor:pointer;
    color:var(--bg); background:var(--fg); border:0; border-radius:10px; }
  button:disabled { opacity:.5; cursor:default; }
  .examples { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:26px; }
  .examples button { background:transparent; color:var(--muted); font-weight:400;
    font-size:13px; padding:5px 10px; border:1px solid var(--line); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:18px 20px; margin-bottom:16px; }
  .badge { display:inline-block; font-size:11px; font-weight:700; letter-spacing:.08em;
    padding:3px 9px; border-radius:999px; border:1px solid currentColor; }
  .GROUNDED { color:var(--accent); } .GENERAL { color:var(--warn); } .REFUSE { color:var(--stop); }
  .meta { color:var(--muted); font-size:12px; margin-left:10px; }
  .answer { white-space:pre-wrap; margin:14px 0 0; }
  .answer table { border-collapse:collapse; margin:10px 0; font-size:14px; }
  .answer td, .answer th { border:1px solid var(--line); padding:5px 9px; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.07em;
       color:var(--muted); margin:20px 0 8px; font-weight:600; }
  ol { margin:0; padding-left:20px; }
  li { margin-bottom:5px; font-size:14px; word-break:break-all; }
  a { color:inherit; }
  .spin { color:var(--muted); font-size:14px; }
  .err { color:var(--stop); }
</style>
</head>
<body>
<div class="wrap">
  <h1>GIKI Assistant</h1>
  <p class="sub">Answers from GIK Institute's public website. Every answer says
     where it came from &mdash; or admits it doesn't know.</p>

  <form id="f">
    <input type="text" id="q" placeholder="Ask about fees, courses, policies&hellip;"
           autocomplete="off" autofocus>
    <button id="go">Ask</button>
  </form>
  <div class="examples" id="ex"></div>

  <div id="out"></div>
</div>
<script>
const EXAMPLES = [
  "What are the undergraduate fee charges?",
  "What are the rules about student discipline?",
  "When does the fall semester start?",
  "What scholarships are available?",
  "How do I deal with exam stress?"
];
const ex = document.getElementById('ex');
EXAMPLES.forEach(t => {
  const b = document.createElement('button');
  b.type = 'button'; b.textContent = t;
  b.onclick = () => { document.getElementById('q').value = t; ask(); };
  ex.appendChild(b);
});

const esc = s => s.replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// Markdown tables survive extraction end-to-end, so render them as tables --
// a fee table read as plain text is exactly the ambiguity we removed upstream.
function renderBody(text) {
  const lines = text.split('\\n');
  let html = '', rows = [];
  const flush = () => {
    if (!rows.length) return;
    const body = rows.filter(r => !/^\\s*\\|[\\s|:-]+\\|\\s*$/.test(r));
    html += '<table>' + body.map((r, i) => {
      const cells = r.trim().replace(/^\\||\\|$/g, '').split('|');
      const tag = i === 0 ? 'th' : 'td';
      return '<tr>' + cells.map(c => `<${tag}>${esc(c.trim())}</${tag}>`).join('') + '</tr>';
    }).join('') + '</table>';
    rows = [];
  };
  for (const line of lines) {
    if (line.trim().startsWith('|')) rows.push(line);
    else { flush(); html += esc(line) + '\\n'; }
  }
  flush();
  return html;
}

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const out = document.getElementById('out'), go = document.getElementById('go');
  go.disabled = true;
  out.innerHTML = '<div class="card spin">Searching the corpus&hellip;</div>';
  try {
    const r = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    let h = `<div class="card"><span class="badge ${d.mode}">${d.mode}</span>`;
    h += `<span class="meta">${d.latency_s.toFixed(1)}s &middot; ${d.chunks_used} sources read &middot; ${esc(d.model)}</span>`;
    h += `<div class="answer">${renderBody(d.text)}</div>`;
    if (d.citations.length) {
      h += '<h2>Sources</h2><ol>' + d.citations.map(c =>
        `<li><a href="${esc(c)}" target="_blank" rel="noopener">${esc(c)}</a></li>`).join('') + '</ol>';
    }
    out.innerHTML = h + '</div>';
  } catch (e) {
    out.innerHTML = `<div class="card err">Something went wrong: ${esc(e.message)}</div>`;
  } finally {
    go.disabled = false;
  }
}
document.getElementById('f').onsubmit = e => { e.preventDefault(); ask(); };
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    # One question at a time per process would serialise the UI; Ollama itself
    # queues, so concurrency here only keeps the page responsive.
    lock = threading.Lock()

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            question = json.loads(self.rfile.read(length)).get("question", "").strip()
            if not question:
                raise ValueError("empty question")
            with self.lock:
                result = ans.answer(question)
            payload = {
                "mode": result.mode,
                "text": result.text,
                "citations": result.citations,
                "latency_s": result.latency_s,
                "chunks_used": result.chunks_used,
                "model": result.model,
            }
        except Exception as exc:  # noqa: BLE001 - report, never crash the server
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        self._send(200, json.dumps(payload).encode("utf-8"), "application/json")

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"GIKI Assistant on http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
