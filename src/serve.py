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
import socket
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer as ans  # noqa: E402
import voice  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

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
  .icon { width:44px; padding:0; font-weight:600; color:var(--fg); background:var(--card);
    border:1px solid var(--line); border-radius:12px; cursor:pointer; font-size:17px; }
  .icon:disabled { opacity:.4; cursor:default; }
  .icon.rec { color:#fff; background:#b5342f; border-color:#b5342f;
    animation:pulse 1.1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.55} }
  .speak { background:none; border:0; color:var(--muted); cursor:pointer; font-size:14px;
    padding:0 4px; margin-left:8px; }
  .speak:hover { color:var(--fg); }
  .toggle { font-size:12.5px; color:var(--muted); display:flex; align-items:center; gap:5px; }
  .hint { max-width:780px; margin:0 auto 8px; font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <span class="dot"></span><h1>GIKI Assistant</h1>
  <span class="sp"></span>
  <label class="toggle"><input type="checkbox" id="auto"> Speak replies</label>
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
  <div class="hint" id="hint"></div>
  <div class="composer">
    <button class="icon" id="mic" title="Hold to talk, click to start/stop">&#127908;</button>
    <textarea id="q" rows="1" placeholder="Ask a question, or tap the mic&hellip;"></textarea>
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
    if (caps.tts) {
      const sp = document.createElement('button');
      sp.className = 'speak'; sp.textContent = '🔊'; sp.title = 'Read aloud';
      sp.onclick = () => speakText(d.text, sp);
      pending.querySelector('.meta').after(sp);
      if (auto.checked) speakText(d.text, sp);
    }

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

// ---------------------------------------------------------------- voice
let caps = {stt:false, tts:false};
let recorder = null, chunks = [], recording = false, levelCtx = null, heard = false;
const mic = document.getElementById('mic');
const auto = document.getElementById('auto');
const hint = document.getElementById('hint');

fetch('/capabilities').then(r => r.json()).then(c => {
  caps = c;
  if (!caps.tts) auto.parentElement.style.display = 'none';
  // getUserMedia only exists in a secure context. Say so plainly rather than
  // letting the mic button fail silently.
  if (!navigator.mediaDevices || !window.isSecureContext) {
    mic.disabled = true;
    hint.textContent = 'Microphone needs HTTPS or localhost — open the https:// address to dictate.';
  }
}).catch(() => {});

// Encode Float32 PCM as 16-bit mono WAV. Done here so the server needs no
// ffmpeg (absent on the box, and sudo is restricted).
function toWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (const s of samples) {
    const c = Math.max(-1, Math.min(1, s));
    v.setInt16(o, c < 0 ? c * 0x8000 : c * 0x7fff, true); o += 2;
  }
  return new Blob([buf], {type: 'audio/wav'});
}

async function startRec() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      mic.classList.remove('rec'); mic.disabled = true;
      hint.textContent = 'Transcribing…';
      try {
        const blob = new Blob(chunks, {type: chunks[0]?.type || 'audio/webm'});
        const ac = new AudioContext();
        const decoded = await ac.decodeAudioData(await blob.arrayBuffer());
        const wav = toWav(decoded.getChannelData(0), decoded.sampleRate);
        await ac.close();
        const r = await fetch('/transcribe', {
          method: 'POST', headers: {'Content-Type': 'audio/wav'}, body: wav});
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        if (d.text) { box.value = (box.value ? box.value + ' ' : '') + d.text; box.focus(); }
        if (d.silent) {
          hint.textContent = `No sound reached the server (${d.seconds}s recorded, `
            + `peak ${d.peak}). Check the browser is using the right microphone, `
            + `that it is not muted, and that this page is open on the device with the mic.`;
        } else {
          hint.textContent = d.text ? '' : "Didn't catch that — try again.";
        }
      } catch (e) {
        hint.textContent = 'Transcription failed: ' + e.message;
      } finally {
        mic.disabled = false;
      }
    };
    // Live level meter. Silence is invisible until it is too late otherwise --
    // you only find out when the transcript comes back wrong.
    try {
      const ac = new AudioContext();
      const src = ac.createMediaStreamSource(stream);
      const an = ac.createAnalyser();
      an.fftSize = 512;
      src.connect(an);
      const buf = new Uint8Array(an.fftSize);
      levelCtx = ac;
      const tick = () => {
        if (!recording) { ac.close().catch(() => {}); levelCtx = null; return; }
        an.getByteTimeDomainData(buf);
        let peak = 0;
        for (const v of buf) peak = Math.max(peak, Math.abs(v - 128) / 128);
        heard = heard || peak > 0.02;
        const bars = '▁▂▃▄▅▆▇█';
        const n = Math.min(bars.length - 1, Math.round(peak * 22));
        hint.textContent = `Listening ${bars[n].repeat(12)}  `
          + (heard ? '' : '(no sound yet — is the right mic selected?)');
        requestAnimationFrame(tick);
      };
      tick();
    } catch (e) { /* meter is a nicety; never block recording */ }

    recorder.start();
    recording = true;
    mic.classList.add('rec');
  } catch (e) {
    hint.textContent = 'Microphone blocked: ' + e.message;
  }
}

mic.onclick = () => {
  if (recording) { recording = false; recorder.stop(); }
  else { heard = false; startRec(); }
};

let audioEl = null;
async function speakText(text, btn) {
  if (audioEl) { audioEl.pause(); audioEl = null; }
  if (btn) btn.textContent = '⏳';
  try {
    const r = await fetch('/speak', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})});
    if (!r.ok) throw new Error('speech unavailable');
    audioEl = new Audio(URL.createObjectURL(await r.blob()));
    audioEl.onended = () => { if (btn) btn.textContent = '🔊'; };
    await audioEl.play();
  } catch (e) {
    if (btn) btn.textContent = '🔇';
  } finally {
    if (btn && btn.textContent === '⏳') btn.textContent = '🔊';
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
        elif self.path == "/capabilities":
            body = json.dumps({"stt": True, "tts": voice.tts_ready()}).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def _transcribe(self) -> None:
        """Browser posts 16 kHz mono WAV; Whisper returns text."""
        length = int(self.headers.get("Content-Length", 0))
        if length > 25 * 1024 * 1024:
            self._send(413, b'{"error":"audio too large"}', "application/json")
            return
        try:
            text = voice.transcribe(self.rfile.read(length))
            body = json.dumps({"text": text}).encode()
        except voice.SilentAudio as quiet:
            # Report it honestly instead of letting Whisper's silence
            # hallucination ("you") reach the user as if it were speech.
            body = json.dumps({
                "text": "",
                "silent": True,
                "peak": round(quiet.peak, 4),
                "seconds": round(quiet.seconds, 1),
            }).encode()
        except Exception as exc:  # noqa: BLE001
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
        self._send(200, body, "application/json")

    def _speak(self) -> None:
        """Text in, WAV out. Empty body means TTS is unavailable, which the UI
        treats as 'hide the speaker button' rather than an error."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            text = str(json.loads(self.rfile.read(length)).get("text", ""))
            audio = voice.speak(text)
        except Exception:  # noqa: BLE001
            audio = b""
        if not audio:
            self._send(503, b"", "audio/wav")
            return
        self._send(200, audio, "audio/wav")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/transcribe":
            with self.lock:
                self._transcribe()
            return
        if self.path == "/speak":
            with self.lock:
                self._speak()
            return
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
    ap.add_argument("--cert", default=str(REPO_ROOT / "certs" / "server.crt"))
    ap.add_argument("--key", default=str(REPO_ROOT / "certs" / "server.key"))
    ap.add_argument("--http", action="store_true",
                    help="serve plain HTTP (microphone will be blocked off localhost)")
    args = ap.parse_args()

    # Two listeners on the same port, one per address family. Both are needed:
    #
    #   IPv4 (0.0.0.0) -- VS Code's port scanner reads /proc/net/tcp, the IPv4
    #     table. A dual-stack IPv6 socket appears only in /proc/net/tcp6, so it
    #     is invisible to auto-forwarding. Every port VS Code did forward here
    #     (ollama, the extension host) is listed in the IPv4 table.
    #   IPv6 ([::], V6ONLY) -- this box's /etc/hosts resolves "localhost" to ::1
    #     before 127.0.0.1, so anything connecting by name lands on IPv6.
    #
    # V6ONLY=1 lets the two coexist rather than fighting over the port.
    def build(family: int, bind: str):
        class Server(ThreadingHTTPServer):
            address_family = family

            def server_bind(self):
                if family == socket.AF_INET6:
                    self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                super().server_bind()

        return Server((bind, args.port), Handler)

    wildcard = args.host in ("0.0.0.0", "::", "")
    plan = ([(socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")] if wildcard
            else [(socket.AF_INET, args.host)])

    servers = []
    for family, bind in plan:
        try:
            servers.append(build(family, bind))
        except OSError as exc:
            print(f"  note: could not bind {bind}:{args.port} ({exc})", flush=True)
    if not servers:
        print(f"error: nothing could bind port {args.port}", flush=True)
        return 1

    scheme = "http"
    if not args.http and Path(args.cert).exists() and Path(args.key).exists():
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        for srv in servers:
            srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = "https"
    elif not args.http:
        print("  no cert found -- serving HTTP; the microphone will only work "
              "via localhost or a forwarded port", flush=True)

    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    server = servers[0]

    print(f"GIKI Assistant on {scheme}://{args.host}:{args.port}  (Ctrl-C to stop)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
