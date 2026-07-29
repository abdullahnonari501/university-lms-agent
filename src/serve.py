"""Phase 4: chat UI over answer.answer().

Standard library only -- no Flask, no FastAPI, nothing to pip install. The box
runs a restricted sudo and is not permanently ours, so a server that needs no
installation and no root is worth more than a nicer framework.

Multi-turn. The server stays stateless: the browser holds the transcript and
sends it with each turn, and answer() folds it into a standalone search query
before retrieval. Retrieval itself has no memory, so the rewrite is what makes
follow-ups work -- storing messages alone would not.

    python3 src/serve.py --host 0.0.0.0        # reachable on the LAN
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer as ans  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GIKI Assistant</title>
<style>
  :root {
    --bg:#faf9f7; --fg:#1c1b19; --muted:#6f6b64; --line:#e4e1db;
    --card:#fff; --mine:#1c1b19; --mineFg:#faf9f7;
    --ok:#1c5d3a; --warn:#8a5a00; --stop:#8c3a3a; --accent:#1c5d3a;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#131217; --fg:#ecebe7; --muted:#9c978f; --line:#2b2932;
            --card:#1c1b22; --mine:#ecebe7; --mineFg:#131217;
            --ok:#7cc79c; --warn:#e2b662; --stop:#e89393; --accent:#7cc79c; }
  }
  * { box-sizing:border-box; }
  html,body { height:100%; }
  body { margin:0; background:var(--bg); color:var(--fg); display:flex; flex-direction:column;
    font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  header { border-bottom:1px solid var(--line); padding:14px 20px; display:flex;
    align-items:center; gap:12px; background:var(--bg); position:sticky; top:0; z-index:5; }
  header h1 { font-size:16px; margin:0; font-weight:650; letter-spacing:-.01em; }
  header .dot { width:7px; height:7px; border-radius:50%; background:var(--ok); }
  header .sp { flex:1; }
  header button { font-size:13px; color:var(--muted); background:none;
    border:1px solid var(--line); border-radius:8px; padding:5px 11px; cursor:pointer; }
  main { flex:1; overflow-y:auto; }
  .thread { max-width:780px; margin:0 auto; padding:24px 20px 8px; }
  .intro { color:var(--muted); font-size:14px; margin-bottom:22px; }
  .intro b { color:var(--fg); font-weight:600; }
  .row { display:flex; margin-bottom:18px; }
  .row.me { justify-content:flex-end; }
  .bubble { max-width:88%; padding:12px 15px; border-radius:14px; }
  .me .bubble { background:var(--mine); color:var(--mineFg); border-bottom-right-radius:4px; }
  .bot .bubble { background:var(--card); border:1px solid var(--line);
    border-bottom-left-radius:4px; width:100%; }
  .badge { display:inline-block; font-size:10px; font-weight:700; letter-spacing:.09em;
    padding:2px 8px; border-radius:999px; border:1px solid currentColor; vertical-align:2px; }
  .GROUNDED{color:var(--ok)} .GENERAL{color:var(--warn)} .REFUSE{color:var(--stop)}
  .meta { color:var(--muted); font-size:11.5px; margin-left:9px; }
  .body { white-space:pre-wrap; margin-top:11px; }
  .body table { border-collapse:collapse; margin:10px 0; font-size:13.5px; display:block;
    overflow-x:auto; max-width:100%; }
  .body td,.body th { border:1px solid var(--line); padding:5px 9px; text-align:left;
    white-space:nowrap; }
  .body th { font-weight:600; }
  details { margin-top:12px; }
  summary { color:var(--muted); font-size:12.5px; cursor:pointer; }
  details ol { margin:8px 0 0; padding-left:20px; }
  details li { font-size:13px; margin-bottom:4px; word-break:break-all; }
  a { color:var(--accent); }
  .rewrite { color:var(--muted); font-size:12.5px; margin-top:9px; font-style:italic; }
  .typing span { display:inline-block; width:6px; height:6px; margin-right:3px; border-radius:50%;
    background:var(--muted); animation:b 1.2s infinite; }
  .typing span:nth-child(2){animation-delay:.2s} .typing span:nth-child(3){animation-delay:.4s}
  @keyframes b { 0%,60%,100%{opacity:.25} 30%{opacity:1} }
  footer { border-top:1px solid var(--line); background:var(--bg);
    position:sticky; bottom:0; padding:12px 20px 16px; }
  .composer { max-width:780px; margin:0 auto; display:flex; gap:9px; }
  textarea { flex:1; resize:none; font:inherit; color:var(--fg); background:var(--card);
    border:1px solid var(--line); border-radius:12px; padding:11px 14px; max-height:140px; }
  textarea:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  .send { padding:0 18px; font-weight:600; color:var(--mineFg); background:var(--mine);
    border:0; border-radius:12px; cursor:pointer; }
  .send:disabled { opacity:.45; cursor:default; }
  .chips { max-width:780px; margin:0 auto 10px; display:flex; flex-wrap:wrap; gap:6px; }
  .chips button { font-size:12.5px; color:var(--muted); background:none;
    border:1px solid var(--line); border-radius:999px; padding:5px 11px; cursor:pointer; }
  .err { color:var(--stop); }
</style>
</head>
<body>
<header>
  <span class="dot"></span><h1>GIKI Assistant</h1>
  <span class="sp"></span>
  <button id="clear">New chat</button>
</header>

<main><div class="thread" id="thread">
  <p class="intro">Ask about GIK Institute &mdash; fees, courses, policies, people.
    Every reply is labelled <b>GROUNDED</b> (from the website, with sources),
    <b>GENERAL</b> (my own knowledge, flagged) or <b>REFUSE</b> (not published &mdash;
    it won't guess). Follow-up questions work; it remembers the conversation.</p>
</div></main>

<footer>
  <div class="chips" id="chips"></div>
  <div class="composer">
    <textarea id="q" rows="1" placeholder="Ask a question&hellip;"></textarea>
    <button class="send" id="go">Send</button>
  </div>
</footer>

<script>
const CHIPS = ["What are the undergraduate fee charges?",
               "What are the rules about student discipline?",
               "What scholarships are available?"];
let history = [];      // [{role, content}] -- the server is stateless
let busy = false;

const thread = document.getElementById('thread');
const box = document.getElementById('q');
const go = document.getElementById('go');

const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

CHIPS.forEach(t => {
  const b = document.createElement('button');
  b.textContent = t.length > 42 ? t.slice(0, 40) + '…' : t;
  b.title = t;
  b.onclick = () => { box.value = t; send(); };
  document.getElementById('chips').appendChild(b);
});

// Markdown tables survive extraction, chunking and retrieval with their columns
// intact; flattening them back to prose here would undo all of that and make a
// fee figure ambiguous again.
function renderBody(text) {
  const out = [];
  let rows = [];
  const flush = () => {
    if (!rows.length) return;
    const body = rows.filter(r => !/^\s*\|[\s|:-]+\|\s*$/.test(r));
    out.push('<table>' + body.map((r, i) => {
      const cells = r.trim().replace(/^\||\|$/g, '').split('|');
      const tag = i === 0 ? 'th' : 'td';
      return '<tr>' + cells.map(c => `<${tag}>${esc(c.trim())}</${tag}>`).join('') + '</tr>';
    }).join('') + '</table>');
    rows = [];
  };
  for (const line of text.split('\n')) {
    if (line.trim().startsWith('|')) rows.push(line);
    else { flush(); out.push(esc(line)); }
  }
  flush();
  return out.join('\n');
}

function addRow(cls, inner) {
  const row = document.createElement('div');
  row.className = 'row ' + cls;
  row.innerHTML = `<div class="bubble">${inner}</div>`;
  thread.appendChild(row);
  row.scrollIntoView({behavior: 'smooth', block: 'end'});
  return row;
}

async function send() {
  const q = box.value.trim();
  if (!q || busy) return;
  busy = true; go.disabled = true;
  box.value = ''; box.style.height = 'auto';

  addRow('me', esc(q));
  const pending = addRow('bot',
    '<span class="typing"><span></span><span></span><span></span></span>');

  try {
    const r = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q, history})
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);

    let h = `<span class="badge ${d.mode}">${d.mode}</span>`;
    h += `<span class="meta">${d.latency_s.toFixed(1)}s · ${d.chunks_used} sources read</span>`;
    h += `<div class="body">${renderBody(d.text)}</div>`;
    // Show the rewrite only when it changed, so a surprising answer is explicable.
    if (d.search_query && d.search_query.toLowerCase() !== q.toLowerCase()) {
      h += `<div class="rewrite">Searched for: “${esc(d.search_query)}”</div>`;
    }
    if (d.citations.length) {
      h += `<details><summary>${d.citations.length} source${d.citations.length>1?'s':''}</summary><ol>`
         + d.citations.map(c => `<li><a href="${esc(c)}" target="_blank" rel="noopener">${esc(c)}</a></li>`).join('')
         + '</ol></details>';
    }
    pending.querySelector('.bubble').innerHTML = h;

    history.push({role: 'user', content: q});
    history.push({role: 'assistant', content: d.text});
    if (history.length > 12) history = history.slice(-12);
  } catch (e) {
    pending.querySelector('.bubble').innerHTML =
      `<span class="err">Couldn't answer: ${esc(e.message)}</span>`;
  } finally {
    busy = false; go.disabled = false; box.focus();
  }
}

go.onclick = send;
box.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
box.addEventListener('input', () => {
  box.style.height = 'auto';
  box.style.height = Math.min(box.scrollHeight, 140) + 'px';
});
document.getElementById('clear').onclick = () => {
  history = [];
  thread.querySelectorAll('.row').forEach(r => r.remove());
  box.focus();
};
box.focus();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
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
            data = json.loads(self.rfile.read(length))
            question = str(data.get("question", "")).strip()
            if not question:
                raise ValueError("empty question")

            # Trust nothing from the browser: keep only well-formed turns, and
            # cap the count so a crafted payload cannot blow the context window.
            history = [
                {"role": t.get("role"), "content": str(t.get("content", ""))[:1500]}
                for t in (data.get("history") or [])
                if isinstance(t, dict) and t.get("role") in ("user", "assistant")
            ][-ans.MAX_HISTORY_TURNS * 2:]

            with self.lock:
                result = ans.answer(question, history=history)
            payload = {
                "mode": result.mode,
                "text": result.text,
                "citations": result.citations,
                "latency_s": result.latency_s,
                "chunks_used": result.chunks_used,
                "model": result.model,
                "search_query": result.search_query,
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
