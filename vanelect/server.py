"""FastAPI backend: serves the frontend, precomputed data, citations, and
on-demand re-projection for custom topic weights / combined block weights.

Embedding credentials (if an API backend is configured) live only in this
process's environment; the browser never sees them and no endpoint embeds
arbitrary user text.
"""
from __future__ import annotations

import html
import json
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .fetch import ROOT
from . import analyze

SITE = ROOT / "data" / "site"
EXP = ROOT / "data" / "exports"
WEB = ROOT / "web"
TEXT = ROOT / "data" / "text"

app = FastAPI(title="Vancouver 2026 candidate explorer", docs_url="/api/docs")


@lru_cache(maxsize=1)
def state() -> analyze.State:
    return analyze.State()


@lru_cache(maxsize=1)
def _json(name: str):
    return json.loads((SITE / name).read_text())


def load(name):
    return json.loads((SITE / name).read_text())


@app.get("/api/meta")
def meta():
    return JSONResponse(load("meta.json"))


@app.get("/api/candidates")
def candidates():
    return JSONResponse(load("candidates.json"))


@app.get("/api/analysis")
def analysis():
    return FileResponse(SITE / "analysis.json", media_type="application/json")


@app.get("/api/sources")
def sources():
    return JSONResponse(load("sources.json"))


@app.get("/api/claims/{cid}")
def claims(cid: str):
    p = SITE / "claims" / f"{cid}.json"
    if not p.resolve().is_relative_to((SITE / "claims").resolve()) or not p.exists():
        return JSONResponse([])
    return FileResponse(p, media_type="application/json")


@app.get("/api/map")
def custom_map(race: str = Query(...), view: str = "policy", mode: str = "individual",
               weights: str = "{}", blocks: str = "{}", missing: str = "shared"):
    """Re-project with user weights. view=combined uses block weights."""
    st = state()
    try:
        w = {k: float(v) for k, v in json.loads(weights).items()}
        b = {k: float(v) for k, v in json.loads(blocks).items()}
    except (ValueError, AttributeError):
        raise HTTPException(400, "weights/blocks must be JSON objects of numbers")
    r = None if race == "all" else race
    if r not in (None, *analyze.RACES):
        raise HTTPException(400, "unknown race")
    if view == "combined":
        if missing not in ("shared", "exclude"):
            raise HTTPException(400, "missing must be shared|exclude")
        return JSONResponse(combined(st, r, mode, w, b, missing))
    if view not in ("policy", "platform", "experience", "actions"):
        raise HTTPException(400, "unknown view")
    return JSONResponse(analyze.analyse_map(st, r, view, mode, weights=w, sensitivity=False))


def combined(st, race, mode, w, b, missing="shared"):
    """Combine the policy / experience / actions blocks with adjustable weights.

    missing="shared" (default): pairwise similarity is the weight-averaged cosine over the blocks BOTH
    candidates have; the map is classical MDS of the resulting distances. The highest-weighted block is
    required so every pair shares at least one block. A missing block never becomes zeros; it simply
    does not enter that pair's comparison.
    missing="exclude": only candidates with every weighted block; L2-normalised blocks scaled by
    sqrt(weight) are concatenated and projected with PCA.
    """
    blocks = {k: v for k, v in b.items() if v > 0 and k in ("policy", "experience", "actions")}
    if not blocks:
        raise HTTPException(400, "give at least one positive block weight")
    anchor = max(blocks, key=lambda k: (blocks[k], k == "policy"))
    per, why = {}, {}
    for blk in blocks:
        ids, X, _, insuff = analyze.candidate_vectors(st, race, blk, mode if blk == "policy" else "individual",
                                                      w if blk != "experience" else None)
        per[blk] = dict(zip(ids, X))
        for c, r in insuff.items():
            why.setdefault(c, []).append(f"{blk}: {r}")
    pool = [c["candidate_id"] for c in st.cands if race is None or c["race"] == race]
    if missing == "exclude":
        ids = [c for c in pool if all(c in per[blk] for blk in blocks)]
    else:
        ids = [c for c in pool if c in per[anchor]]
    res = dict(race=race, view="combined", mode=mode, blocks=blocks, missing=missing, anchor=anchor, n=len(ids),
               insufficient={c: "; ".join(why.get(c, ["no evidence in a required block"])) for c in pool if c not in ids},
               blocks_present={c: [blk for blk in blocks if c in per[blk]] for c in ids})
    if missing == "exclude":
        res["note"] = ("Combined map (strict): each block L2-normalised and scaled by sqrt(weight), concatenated, PCA. "
                       "Candidates missing any weighted block are excluded, never zero-filled.")
    else:
        res["note"] = (f"Combined map (shared blocks): similarity between two candidates is the weight-averaged cosine "
                       f"over the blocks both have; the {anchor} block is required so every pair shares one. Missing "
                       f"blocks are never zero-filled. Map = classical MDS of sqrt(2 - 2 x similarity).")
    if len(ids) < 3:
        res["points"] = [dict(id=c) for c in ids]
        return res
    n = len(ids)
    if missing == "exclude":
        X = np.hstack([np.array([per[blk][c] for c in ids]) * np.sqrt(wt) for blk, wt in blocks.items()])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y, evr = analyze.pca2(X)
        S = X @ X.T
        res["pca"] = dict(explained_variance_ratio=evr)
    else:
        num = np.zeros((n, n))
        den = np.zeros((n, n))
        for blk, wt in blocks.items():
            have = np.array([c in per[blk] for c in ids])
            V = np.array([per[blk][c] if c in per[blk] else np.zeros(st.vecs.shape[1]) for c in ids])
            both = np.outer(have, have)
            num += wt * (V @ V.T) * both
            den += wt * both
        S = num / den  # anchor guarantees den > 0
        D2 = np.clip(2 - 2 * S, 0, None)
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        ev, U = np.linalg.eigh(B)
        o = np.argsort(ev)[::-1]
        ev, U = ev[o], U[:, o]
        pos = np.clip(ev, 0, None)
        Y = U[:, :2] * np.sqrt(pos[:2])
        for k in range(2):
            if Y[np.argmax(np.abs(Y[:, k])), k] < 0:
                Y[:, k] *= -1
        neg = float(-ev[ev < 0].sum() / np.abs(ev).sum()) if np.abs(ev).sum() else 0.0
        res["pca"] = dict(explained_variance_ratio=[round(float(x), 4) for x in (pos / pos.sum())[:10]],
                          method="classical MDS", negative_eigen_share=round(neg, 4))
    np.fill_diagonal(S, -np.inf)
    nb = [list(np.argsort(-S[i])[:min(5, n - 1)]) for i in range(n)]
    res["umap"] = dict(available=False, note="UMAP not computed for custom combined maps")
    res["points"] = [dict(id=c, pca=[round(float(v), 5) for v in Y[i]], umap=None, blocks=res["blocks_present"][c],
                          neighbours=[dict(id=ids[j], cosine=round(float(S[i, j]), 4)) for j in nb[i]])
                     for i, c in enumerate(ids)]
    return res


@app.get("/api/support")
def support(view: str, a: str, b: str, mode: str = "individual"):
    st = state()
    if a not in st.cmap or b not in st.cmap:
        raise HTTPException(404)
    return JSONResponse(analyze.support_pairs(st, view, a, b, mode, top=3))


@app.get("/source/{sid}", response_class=HTMLResponse)
def source_view(sid: str, p: str | None = None):
    """Cached, extracted text of a source with passage anchors (citation target)."""
    srcs = {s["source_id"]: s for s in load("sources.json")}
    s = srcs.get(sid)
    path = TEXT / f"{sid}.txt"
    if not s or not path.exists():
        raise HTTPException(404, "unknown source")
    body = []
    for block in path.read_text().split("\n\n"):
        if block.startswith("# "):
            continue
        if block.startswith("## "):
            body.append(f"<h3>{html.escape(block[3:])}</h3>")
            continue
        if block.startswith("[") and "] " in block:
            pid, txt = block[1:].split("] ", 1)
            cls = " class=hit" if pid == p else ""
            body.append(f"<p id='{html.escape(pid)}'{cls}><a class=pid href='#{html.escape(pid)}'>{html.escape(pid.split(':', 1)[1])}</a> "
                        f"{html.escape(txt)}</p>")
    via = s.get("via")
    arch = (f" via Internet Archive snapshot <a href='https://web.archive.org/web/{s['wayback_ts']}/{html.escape(s['url'])}'>"
            f"{s['wayback_ts']}</a>") if via == "wayback" and s.get("wayback_ts") else ""
    head = (f"<h1>{html.escape(s.get('title') or s['url'])}</h1><p class=meta>{html.escape(s.get('publisher') or '')} · "
            f"{html.escape(s.get('published') or 'date not stated')} · {html.escape(s.get('source_type') or '')}<br>"
            f"Original: <a href='{html.escape(s['url'])}' rel=noopener target=_blank>{html.escape(s['url'])}</a>{arch}<br>"
            f"Retrieved {html.escape(s.get('retrieved_at') or '')} · sha256 <code>{html.escape((s.get('content_hash') or '')[:20])}…</code></p>"
            + (f"<p class=persp><b>Publisher perspective:</b> {html.escape(s['perspective'])}</p>" if s.get("perspective") else ""))
    css = ("body{font:16px/1.55 system-ui,sans-serif;max-width:820px;margin:1.5rem auto;padding:0 16px;background:#fff;color:#1b1b1b}"
           ".meta{color:#555;font-size:14px}p.hit{background:#fff3b0;outline:3px solid #fff3b0}.pid{font:12px monospace;color:#888;"
           "text-decoration:none;margin-right:.4em}.persp{background:#f2f4f8;padding:8px}@media(prefers-color-scheme:dark){body{background:#141414;"
           "color:#e6e6e6}.meta{color:#aaa}p.hit{background:#5a4c00;outline-color:#5a4c00}.persp{background:#22252b}}")
    js = "<script>const h=document.querySelector('.hit');if(h)h.scrollIntoView({block:'center'});</script>"
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Source {sid}</title><style>{css}</style>{head}{''.join(body)}{js}")


# ------------------------------------------------------------------ browser captures (bookmarklet)
from fastapi import Request  # noqa: E402
import secrets as _secrets  # noqa: E402
from . import captures  # noqa: E402


def _check_token(t: str | None) -> None:
    tok = captures.token()
    if not tok or not t or not _secrets.compare_digest(t, tok):
        raise HTTPException(403, "invalid token")


@app.post("/api/ingest")
async def ingest(request: Request, t: str | None = None):
    _check_token(t)
    body = await request.body()
    if len(body) > 60 * 1024 * 1024:
        raise HTTPException(413, "too large")
    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(400, "body must be JSON")
    return JSONResponse(captures.ingest(payload))


@app.get("/contribute", response_class=HTMLResponse)
def contribute(request: Request, t: str | None = None, recv: int = 0):
    _check_token(t)
    origin = f"https://{request.headers.get('host', 'vanelection2026.help')}"
    bm = ("javascript:(async()=>{const O=" + json.dumps(origin) + ",T=" + json.dumps(t) + ";"
          "const w=window.open(O+'/contribute?recv=1&t='+encodeURIComponent(T),'vanelect','width=560,height=460');"
          "const pdfs=[...new Set([...document.querySelectorAll('a[href]')].map(a=>a.href)"
          ".filter(h=>/\\.pdf($|[?#])/i.test(h)&&/candidates-list|minutes|voting-record/i.test(h)))].slice(0,30);"
          "const files=[];for(const u of pdfs){try{const r=await fetch(u,{credentials:'include'});if(!r.ok)continue;"
          "const b=new Uint8Array(await r.arrayBuffer());let s='';for(let i=0;i<b.length;i+=32768)"
          "s+=String.fromCharCode.apply(null,b.subarray(i,i+32768));files.push({url:u,b64:btoa(s)})}catch(e){}}"
          "if(/\\.pdf($|[?#])/i.test(location.href)){try{const r=await fetch(location.href,{credentials:'include'});"
          "const b=new Uint8Array(await r.arrayBuffer());let s='';for(let i=0;i<b.length;i+=32768)"
          "s+=String.fromCharCode.apply(null,b.subarray(i,i+32768));files.push({url:location.href,b64:btoa(s)})}catch(e){}}"
          "const m={url:location.href,title:document.title,html:document.contentType==='text/html'?"
          "document.documentElement.outerHTML:'',captured_at:new Date().toISOString(),user_agent:navigator.userAgent,files};"
          "let n=0;const iv=setInterval(()=>{w.postMessage(m,O);if(++n>60)clearInterval(iv)},500);"
          "addEventListener('message',e=>{if(e.origin===O&&e.data==='ack')clearInterval(iv)})})()")
    caps = captures.all_captures()
    rows = "".join(f"<li>{html.escape(c['captured_at'][:16])} · {c['kind']} · {html.escape(c['url'])} "
                   f"<span class=m>{c['bytes']:,} bytes · sha256 {c['sha256'][:10]}</span></li>" for c in caps[::-1][:50])
    css = ("body{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:1.5rem auto;padding:0 16px;background:#fff;"
           "color:#1b1b1b}a.bm{display:inline-block;padding:.5rem .9rem;border-radius:6px;background:#2a78d6;color:#fff;"
           "text-decoration:none;font-weight:600}.m{color:#777;font-size:.85em}code{background:#f3f3f3;padding:0 3px}"
           "#st{padding:.6rem;border-radius:6px;background:#f2f4f8;white-space:pre-wrap}"
           "@media(prefers-color-scheme:dark){body{background:#141414;color:#e6e6e6}code,#st{background:#262626}}")
    if recv:
        js = ("<script>const T=" + json.dumps(t) + ";let done=false;addEventListener('message',async e=>{"
              "let h='';try{h=new URL(e.origin).hostname}catch(_){}"
              "if(done||!(h==='vancouver.ca'||h.endsWith('.vancouver.ca')))return;done=true;"
              "e.source.postMessage('ack',e.origin);const st=document.getElementById('st');st.textContent='Uploading…';"
              "try{const r=await fetch('/api/ingest?t='+encodeURIComponent(T),{method:'POST',body:JSON.stringify(e.data)});"
              "const j=await r.json();st.textContent=(j.stored||[]).map(x=>'✓ '+x.kind+'  '+x.url).join('\\n')+"
              "((j.rejected||[]).length?'\\n'+j.rejected.map(x=>'✗ '+x.url+' ('+x.reason+')').join('\\n'):'')||'Nothing stored';"
              "}catch(err){st.textContent='Failed: '+err}});</script>")
        return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
                f"<title>Capture</title><style>{css}</style><h2>Receiving capture…</h2><div id=st>Waiting for the page…</div>"
                f"<p class=m>You can close this window when it shows ✓.</p>{js}")
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Contribute captures</title><style>{css}</style><h1>Add vancouver.ca pages</h1>"
            f"<p>vancouver.ca refuses automated clients, so the pipeline cannot fetch it. You can: open the page in your "
            f"own browser and send what you see.</p><ol><li>Drag this button to your bookmarks bar: "
            f"<a class=bm href=\"{html.escape(bm)}\">Send to Vancouver explorer</a></li>"
            f"<li>Open a vancouver.ca page (e.g. <a href='https://vancouver.ca/your-government/2026-candidates-councillor.aspx' "
            f"target=_blank>council candidates</a>) and click the bookmark. A small window confirms what was stored. Linked "
            f"candidate-list / minutes / voting-record PDFs on that page are sent too; opening a PDF and clicking the "
            f"bookmark sends that PDF.</li><li>Ask Claude to rebuild. Captures replace Internet Archive copies and are "
            f"labelled “captured in a user's browser” with time and hash.</li></ol>"
            f"<p class=m>Only https URLs on vancouver.ca, council.vancouver.ca and parkboardmeetings.vancouver.ca are accepted. "
            f"This page and the bookmark contain a private token: don't share them.</p>"
            f"<h3>Captures so far ({len(caps)})</h3><ul>{rows or '<li>none yet</li>'}</ul>")


@app.get("/methods", response_class=HTMLResponse)
def methods():
    return FileResponse(EXP / "methods_report.html")


app.mount("/exports", StaticFiles(directory=EXP), name="exports")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
