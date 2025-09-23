#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chat_server.py — OpenAI Chat + RAG (separate from app.py)
- Keeps Vector Store API on :8003 intact; this runs on :8004 by default
- Loads .env automatically; configurable via env vars
- Endpoints:
    GET  /health              – health check
    POST /api/chat            – one-shot answer (non-stream)
    GET  /api/chat/stream     – Server-Sent Tokens (streaming answer)
    POST /api/upload          – proxy to VS /api/vs/index (file upload)
    GET  /chat                – minimal UI with streaming + uploads
- Auth: if CHAT_API_SECRET is set → require header X-LOT-KEY on /api/*
- CORS: CORS_ALLOW_ORIGINS (comma-separated), or "*" when empty
"""
from __future__ import annotations

import os, json, time, traceback
from typing import Any, Dict, List, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# ---- .env loader (best-effort)
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    base = Path(__file__).resolve().parent
    for cand in (base/".env", base.parent/".env"):
        if cand.exists():
            load_dotenv(cand, override=False)
            break
except Exception:
    pass

# ---- Third-party clients
try:
    from openai import OpenAI  # pip install openai
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import requests  # pip install requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# ---- ENV & defaults
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
CHAT_MODEL       = os.getenv("CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
CHAT_USE_RESP    = os.getenv("CHAT_USE_RESPONSES", "1").strip() not in ("", "0", "false", "False")
PORT             = int(os.getenv("CHAT_PORT", os.getenv("PORT", "8004")))
VS_API_BASE      = os.getenv("VS_API_BASE", "http://localhost:8003").rstrip("/")
VS_API_KEY       = os.getenv("LOT_API_SECRET", "").strip()
CHAT_API_SECRET  = os.getenv("CHAT_API_SECRET", "").strip()
CORS_ALLOW       = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO").upper()
TIMEOUT_S        = int(os.getenv("TIMEOUT_S", "60"))
TOP_K_DEFAULT    = int(os.getenv("TOP_K", "5"))
THRESH_DEFAULT   = float(os.getenv("SIM_THRESHOLD", "0.2"))

_client: Optional[OpenAI] = None

def client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    global _client
    if _client is None:
        if OpenAI is None:
            raise RuntimeError("Missing 'openai' package")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

# ---- logging & cors

def log(*args):
    if LOG_LEVEL in ("DEBUG", "INFO"):
        print("[CHAT]", *args, flush=True)

def allow_origin(origin: Optional[str]) -> Optional[str]:
    if not CORS_ALLOW:
        return "*"
    if origin and origin in CORS_ALLOW:
        return origin
    return None

# ---- RAG helpers

def rag_search(query: str, top_k: int, threshold: float, include_text: bool = True, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if requests is None:
        return []
    url = VS_API_BASE + "/api/vs/search"
    headers = {"Content-Type": "application/json"}
    if VS_API_KEY:
        headers["X-LOT-KEY"] = VS_API_KEY
    body = {
        "query": query,
        "top_k": int(top_k),
        "threshold": float(threshold),
        "include_text": bool(include_text),
        "filters": filters or {},
    }
    # retry x2
    last_err = None
    for _ in range(2):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=TIMEOUT_S)
            r.raise_for_status()
            data = r.json()
            if data.get("ok"):
                return data.get("results", [])
            last_err = data
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(0.3)
    log("RAG search failed:", last_err)
    return []

SYSTEM_HE = """אתה מומחה רב-תחומי במקרקעין עבור LOTL.
התאם את הכובע המקצועי לתוכן השאלה:
- שמאי מקרקעין: הערכות שווי, עסקאות השוואה, תקינה שמאית.
- אדריכל/מתכנן: ייעודי קרקע, קווי בניין, זכויות בנייה, תב"עות.
- עו"ד מקרקעין/תכנון: היתרים, חריגות, הליכים סטטוטוריים, סיכונים משפטיים.
- מודד/קדסטר: גושים/חלקות, מדידות, מפות.
- מיסוי מקרקעין: מס רכישה, שבח, היטלי השבחה.
כל תשובה תישען על מקורות שנשלפו מהמאגרים (RAG). כאשר אין ראיה מספקת, כתוב: 'על פי הידוע לי כעת, ייתכן חוסר או שינוי במידע'.
אל תמציא עובדות. כתוב עברית מקצועית ותמציתית, והוסף בסוגריים שמות קבצים/מסמכים כמקורות היכן שרלוונטי.
הוסף 'הסתייגויות' בסוף במקרה של אי-ודאות או שונות בין מקורות.
אין לראות באמור ייעוץ משפטי/שמאות מחייב."""

TEMPLATE_HE = """שאלה: {question}

מקטעי רקע רלוונטיים (עד {n}):
{context}

הנחיות ניסוח:
- ענה בתמצית, במבנה נקודתי ברור.
- ציין מקורות בקצרה (שם קובץ/מסמך).
- אם יש פערים/סתירות במקורות – הדגש והצע דרך בדיקה.
"""
def build_context(results: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    if not results:
        return "[אין הקשר ממקורות]"
    used, out = 0, []
    for r in results:
        fn = r.get("filename") or r.get("doc_id") or "source"
        ex = (r.get("excerpt") or "").strip()
        chunk = f"[מקור: {fn}]
{ex}
"
        if used + len(chunk) > max_chars:
            break
        out.append(chunk)
        used += len(chunk)
    return "
".join(out)


def compose_messages(question: str, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    ctx = build_context(results)
    prompt = TEMPLATE_HE.format(question=question, n=len(results), context=ctx)
    return [
        {"role": "system", "content": SYSTEM_HE},
        {"role": "user", "content": prompt},
    ]

# ---- LLM calls: Responses API (default) with fallback to Chat Completions

def llm_answer(question: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    msgs = compose_messages(question, results)
    usage = None
    try:
        if CHAT_USE_RESP:
            # Responses API: single string input built from messages
            text_input = (msgs[0]["content"] + "

" + msgs[1]["content"]).strip()
            resp = client().responses.create(
                model=CHAT_MODEL,
                input=text_input,
                temperature=0.2,
            )
            txt = (resp.output_text or "").strip()
            try:
                usage = getattr(resp, "usage", None)
                usage = {"prompt": getattr(usage, "prompt_tokens", None), "completion": getattr(usage, "completion_tokens", None)} if usage else None
            except Exception:
                usage = None
        else:
            raise RuntimeError("FORCE_FALLBACK")
    except Exception:  # fallback to Chat Completions
        c = client().chat.completions.create(model=CHAT_MODEL, messages=msgs, temperature=0.2)
        txt = c.choices[0].message.content.strip()
        u = getattr(c, "usage", None)
        usage = {"prompt": getattr(u, "prompt_tokens", None), "completion": getattr(u, "completion_tokens", None)} if u else None

    sources = [{
        "doc_id": r.get("doc_id"),
        "filename": r.get("filename"),
        "score": r.get("score"),
        "excerpt": r.get("excerpt"),
        "tags": r.get("tags"),
    } for r in results]
    return {"ok": True, "answer": txt, "sources": sources, "tokens": usage}


def llm_stream(question: str, results: List[Dict[str, Any]]):
    """Generator that yields UTF-8 bytes for streaming endpoint.
    Strategy: send a JSON line with meta (sources) first, then stream tokens.
    Uses Chat Completions stream for robustness.
    """
    # First: meta line with sources
    meta = [{"doc_id": r.get("doc_id"), "filename": r.get("filename"), "score": r.get("score")} for r in results]
    yield (json.dumps({"type": "meta", "sources": meta}, ensure_ascii=False) + "
").encode("utf-8")

    msgs = compose_messages(question, results)
    try:
        stream = client().chat.completions.create(model=CHAT_MODEL, messages=msgs, temperature=0.2, stream=True)
        for chunk in stream:
            delta = getattr(getattr(chunk, "choices", [None])[0], "delta", None)
            if not delta:
                continue
            txt = getattr(delta, "content", None)
            if txt:
                yield txt.encode("utf-8")
        yield b"
"
    except Exception as e:  # send error line
        yield ("
" + json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False) + "
").encode("utf-8")

# ---- HTTP server
class Handler(BaseHTTPRequestHandler):
    server_version = "LOTL-CHAT/1.2"

    def _send_headers(self, status=200, content_type="application/json", extra: Optional[Dict[str,str]] = None):
        self.send_response(status)
        origin = self.headers.get("Origin")
        allow = allow_origin(origin)
        if allow:
            self.send_header("Access-Control-Allow-Origin", allow)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-LOT-KEY")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Type", content_type)
        if extra:
            for k,v in extra.items():
                self.send_header(k, v)
        self.end_headers()

    def _send_json(self, obj: Dict[str, Any], status=200):
        self._send_headers(status, "application/json")
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _auth_ok(self) -> bool:
        if not CHAT_API_SECRET:
            return True
        return self.headers.get("X-LOT-KEY", "") == CHAT_API_SECRET

    # ---- routes
    def do_OPTIONS(self):
        self._send_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return self._send_json({"ok": True, "time": int(time.time()), "port": PORT, "model": CHAT_MODEL, "vs": VS_API_BASE})
        if parsed.path == "/api/chat/stream":
            if not self._auth_ok():
                return self._send_json({"error": "unauthorized"}, status=401)
            qs = parse_qs(parsed.query)
            q = (qs.get("q", [""])[0]).strip()
            top_k = int(qs.get("top_k", [str(TOP_K_DEFAULT)])[0])
            thr = float(qs.get("threshold", [str(THRESH_DEFAULT)])[0])
            results = rag_search(q, top_k=top_k, threshold=thr, include_text=True)
            self._send_headers(200, "text/plain; charset=utf-8", {"Connection": "keep-alive"})
            for chunk in llm_stream(q, results):
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except BrokenPipeError:
                    break
            return
        if parsed.path == "/chat":
            self._send_headers(200, "text/html; charset=utf-8")
            self.wfile.write(CHAT_HTML.encode("utf-8"))
            return
        return self._send_json({"error": "not_found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            if not self._auth_ok():
                return self._send_json({"error": "unauthorized"}, status=401)
            try:
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n) if n else b"{}"
                body = json.loads(raw.decode("utf-8")) if raw else {}
                q = (body.get("message") or "").strip()
                if not q:
                    return self._send_json({"ok": True, "answer": "שאלה ריקה.", "sources": []})
                top_k = int(body.get("top_k", TOP_K_DEFAULT))
                thr = float(body.get("threshold", THRESH_DEFAULT))
                filters = body.get("filters") or {}
                results = rag_search(q, top_k=top_k, threshold=thr, include_text=True, filters=filters)
                return self._send_json(llm_answer(q, results))
            except Exception as e:
                traceback.print_exc()
                return self._send_json({"ok": False, "error": str(e)}, status=500)
        if parsed.path == "/api/upload":
            # simple proxy to VS /api/vs/index (multipart) – expects same form fields
            if requests is None:
                return self._send_json({"ok": False, "error": "requests not installed"}, status=500)
            try:
                # read raw body and forward as-is
                n = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(n) if n else b""
                headers = {k: v for k, v in self.headers.items()}
                headers.pop("Host", None)
                if VS_API_KEY:
                    headers["X-LOT-KEY"] = VS_API_KEY
                r = requests.post(VS_API_BASE + "/api/vs/index", data=body, headers=headers, timeout=TIMEOUT_S)
                return self._send_json(r.json(), status=r.status_code)
            except Exception as e:
                return self._send_json({"ok": False, "error": str(e)}, status=500)
        return self._send_json({"error": "not_found"}, status=404)


CHAT_HTML = """<!doctype html>
<html lang=he dir=rtl>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LOTL Chat</title>
  <style>
    body{font-family:system-ui,Arial;max-width:940px;margin:2rem auto;padding:0 1rem}
    .row{display:flex;gap:.5rem;align-items:center}
    textarea{width:100%;height:120px}
    button{padding:.6rem 1.1rem}
    .msg{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;padding:.75rem;border-radius:.5rem}
    .src{font-size:.9rem;color:#333;margin-top:.5rem}
    .pill{display:inline-block;background:#eef;border:1px solid #dde;padding:.15rem .5rem;border-radius:999px;margin:.15rem}
    .muted{color:#666}
  </style>
</head>
<body>
  <h2>LOTL – צ'אט (RAG)</h2>
  <div class="row">
    <textarea id="q" placeholder="מה השאלה?"></textarea>
  </div>
  <div class="row" style="margin-top:.5rem">
    <button id="ask">שאל</button>
    <button id="askStream">שאל (Streaming)</button>
    <label>top_k <input id="k" type="number" value="5" min="1" max="20" style="width:64px"></label>
    <label>threshold <input id="thr" type="number" step="0.05" value="0.2" style="width:90px"></label>
  </div>
  <div class="row" style="margin-top:.5rem">
    <input id="files" type="file" multiple />
    <button id="upload">העלה מסמכים</button>
    <span id="uStat" class="muted"></span>
  </div>
  <div id="sources" style="margin-top:1rem"></div>
  <div id="out" style="margin-top:.5rem" class="msg"></div>

<script>
const out = document.getElementById('out');
const srcBox = document.getElementById('sources');
const k = document.getElementById('k');
const thr = document.getElementById('thr');

function renderSources(sources){
  if(!sources || !sources.length){ srcBox.innerHTML = '<div class="muted">(אין מקורות)</div>'; return; }
  srcBox.innerHTML = sources.map(s=>`<span class="pill" title="score ${s.score?.toFixed?.(3)??''}">${s.filename||s.doc_id||'מקור'}</span>`).join(' ');
}

async function askOnce(){
  const q = document.getElementById('q').value.trim();
  if(!q){ out.textContent = 'שאלה ריקה'; return; }
  out.textContent = '...'; srcBox.innerHTML = '';
  const body = {message:q, top_k:Number(k.value||5), threshold:Number(thr.value||0.2)};
  const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  if(!j.ok){ out.textContent = 'שגיאה: '+(j.error||'error'); return; }
  out.textContent = j.answer || '';
  renderSources(j.sources);
}

async function askStream(){
  const q = document.getElementById('q').value.trim();
  if(!q){ out.textContent = 'שאלה ריקה'; return; }
  out.textContent=''; srcBox.innerHTML='';
  const params = new URLSearchParams({q, top_k: String(Number(k.value||5)), threshold: String(Number(thr.value||0.2))});
  const r = await fetch('/api/chat/stream?'+params.toString());
  const reader = r.body.getReader();
  let metaShown=false;
  while(true){
    const {done, value} = await reader.read();
    if(done) break;
    const chunk = new TextDecoder().decode(value);
    // first line can be JSON meta with sources
    if(!metaShown){
      const nl = chunk.indexOf('
');
      if(nl>0){
        const first = chunk.slice(0,nl);
        try{ const m = JSON.parse(first); if(m.type==='meta'){ renderSources(m.sources); metaShown=true; out.textContent += chunk.slice(nl+1); continue; } }catch{}
      }
      metaShown=true;
    }
    out.textContent += chunk;
  }
}

async function upload(){
  const fs = document.getElementById('files').files;
  if(!fs || !fs.length){ return; }
  const fd = new FormData();
  for(const f of fs){ fd.append('files[]', f, f.name); }
  fd.append('tags','chat-upload'); fd.append('source','chat');
  document.getElementById('uStat').textContent = 'מעלה...';
  const r = await fetch('/api/upload', {method:'POST', body: fd});
  const j = await r.json();
  document.getElementById('uStat').textContent = j.ok ? 'הועלה' : ('שגיאה: '+(j.error||'error'));
}

document.getElementById('ask').onclick = askOnce;
document.getElementById('askStream').onclick = askStream;
document.getElementById('upload').onclick = upload;
</script>
</body>
</html>"""


def main():
    if not OPENAI_API_KEY:
        print("[CHAT] WARNING: OPENAI_API_KEY not set")
    if requests is None:
        print("[CHAT] WARNING: missing 'requests' package")
    httpd = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f"[CHAT] running on :{PORT} — VS: {VS_API_BASE} — model: {CHAT_MODEL}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()
