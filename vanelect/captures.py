"""Pages captured by a person's own browser (bookmarklet) for sites that refuse automated access.

vancouver.ca serves a Cloudflare bot challenge to scripts. A human reading the
page in a normal browser can pass it on to the pipeline; that is the person
retrieving a public page, not a bot evading a block. Each capture is stored
with the URL, capture time, submitting browser's user agent and SHA-256, and
is labelled `via = "user_browser_capture"` everywhere it is cited.

Only URLs on ALLOWED_HOSTS are accepted.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.parse
from pathlib import Path

from .fetch import ROOT, now_iso, url_key

DIR = ROOT / "data" / "user_captures"
TOKEN_FILE = ROOT / "data" / "ingest_token"
ALLOWED_HOSTS = ("vancouver.ca", "council.vancouver.ca", "parkboardmeetings.vancouver.ca")
MAX_BYTES = 25 * 1024 * 1024


def token() -> str | None:
    return TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None


def allowed(url: str) -> bool:
    u = urllib.parse.urlparse(url)
    host = u.netloc.lower().split(":")[0]
    return u.scheme == "https" and any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


def canonical(url: str) -> str:
    u = urllib.parse.urlparse(url)
    q = "&".join(p for p in u.query.split("&") if p and not p.startswith("utm_"))
    return urllib.parse.urlunparse((u.scheme, u.netloc.lower(), u.path, "", q, ""))


def store(url: str, data: bytes, kind: str, meta: dict) -> dict:
    url = canonical(url)
    d = DIR / url_key(url)
    d.mkdir(parents=True, exist_ok=True)
    ext = "pdf" if kind == "pdf" else "html"
    (d / f"raw.{ext}").write_bytes(data)
    rec = dict(url=url, kind=kind, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
               captured_at=meta.get("captured_at") or now_iso(), received_at=now_iso(),
               user_agent=(meta.get("user_agent") or "")[:300], page_title=(meta.get("title") or "")[:300],
               via="user_browser_capture", path=str((d / f"raw.{ext}").relative_to(ROOT)))
    (d / "meta.json").write_text(json.dumps(rec, indent=1))
    with (DIR / "log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def ingest(payload: dict) -> dict:
    """payload: {url, html, title, captured_at, user_agent, files: [{url, b64}]}"""
    out, rejected = [], []
    url = payload.get("url") or ""
    html = payload.get("html") or ""
    if allowed(url) and html:
        b = html.encode("utf-8")
        if len(b) <= MAX_BYTES and not re.search(r"<title>\s*Just a moment", html[:3000], re.I):
            out.append(store(url, b, "html", payload))
        else:
            rejected.append(dict(url=url, reason="too large or a bot-challenge page"))
    elif url:
        rejected.append(dict(url=url, reason="host not allowed"))
    for f in payload.get("files") or []:
        fu = f.get("url") or ""
        try:
            data = base64.b64decode(f.get("b64") or "", validate=True)
        except ValueError:
            rejected.append(dict(url=fu, reason="bad encoding"))
            continue
        if not allowed(fu) or not data.startswith(b"%PDF") or len(data) > MAX_BYTES:
            rejected.append(dict(url=fu, reason="not an allowed PDF"))
            continue
        out.append(store(fu, data, "pdf", payload))
    return dict(stored=[dict(url=r["url"], kind=r["kind"], bytes=r["bytes"], sha256=r["sha256"]) for r in out],
                rejected=rejected)


def lookup(url: str) -> dict | None:
    m = DIR / url_key(canonical(url)) / "meta.json"
    return json.loads(m.read_text()) if m.exists() else None


def all_captures() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(DIR.glob("*/meta.json"))] if DIR.exists() else []
