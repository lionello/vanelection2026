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
               weights: str = "{}", blocks: str = "{}"):
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
        return JSONResponse(combined(st, r, mode, w, b))
    if view not in ("policy", "platform", "experience", "actions"):
        raise HTTPException(400, "unknown view")
    return JSONResponse(analyze.analyse_map(st, r, view, mode, weights=w, sensitivity=False))


def combined(st, race, mode, w, b):
    """Concatenate L2-normalised block vectors scaled by sqrt(weight): each block's
    influence depends on its weight, not its dimensionality."""
    blocks = {k: v for k, v in b.items() if v > 0 and k in ("policy", "experience", "actions")}
    if not blocks:
        raise HTTPException(400, "give at least one positive block weight")
    per, missing = {}, {}
    for blk in blocks:
        ids, X, _, insuff = analyze.candidate_vectors(st, race, blk, mode if blk == "policy" else "individual",
                                                      w if blk != "experience" else None)
        per[blk] = dict(zip(ids, X))
        for c, why in insuff.items():
            missing.setdefault(c, []).append(f"{blk}: {why}")
    pool = [c["candidate_id"] for c in st.cands if race is None or c["race"] == race]
    ids = [c for c in pool if all(c in per[blk] for blk in blocks)]
    for c in pool:
        if c not in ids and c not in missing:
            missing[c] = ["missing a block"]
    res = dict(race=race, view="combined", mode=mode, blocks=blocks, n=len(ids),
               insufficient={c: "; ".join(v) for c, v in missing.items() if c not in ids},
               note="Combined map: each block L2-normalised and scaled by sqrt(weight) before concatenation. "
                    "Candidates missing any weighted block are excluded, never zero-filled.")
    if len(ids) < 3:
        res["points"] = [dict(id=c) for c in ids]
        return res
    X = np.hstack([np.array([per[blk][c] for c in ids]) * np.sqrt(wt) for blk, wt in blocks.items()])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y, evr = analyze.pca2(X)
    nb = analyze.knn(X, min(5, len(ids) - 1))
    S = X @ X.T
    res["pca"] = dict(explained_variance_ratio=evr)
    res["umap"] = dict(available=False, note="UMAP not computed for custom combined maps")
    res["points"] = [dict(id=c, pca=[round(float(v), 5) for v in Y[i]], umap=None,
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


@app.get("/methods", response_class=HTMLResponse)
def methods():
    return FileResponse(EXP / "methods_report.html")


app.mount("/exports", StaticFiles(directory=EXP), name="exports")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
