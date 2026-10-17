"""Polite, cached HTTP fetching with provenance.

Every fetch is cached under data/cache/<key>/ as:
  raw.<ext>     original bytes (HTML / PDF / JSON)
  meta.json     url, final_url, via (direct|wayback), wayback timestamp,
                status, content_type, sha256, retrieved_at, publisher
Failures (403 bot challenge, 401, DNS, timeouts) are recorded in
data/access_log.jsonl and never retried in a way that tries to defeat an
access restriction: a Cloudflare challenge is logged as `blocked_bot_challenge`
and the only fallback is a public Internet Archive snapshot, if one exists.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
ACCESS_LOG = ROOT / "data" / "access_log.jsonl"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0 Safari/537.36 vancouver-election-explorer/0.1 (research; agent@lunesu.com)")

_last_hit: dict[str, float] = {}
MIN_INTERVAL = {"web.archive.org": 2.5, "archive.org": 2.5}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


@dataclass
class Fetched:
    url: str
    final_url: str
    via: str
    status: int
    content_type: str
    sha256: str
    retrieved_at: str
    path: str
    wayback_ts: str | None = None
    note: str | None = None

    def bytes(self) -> bytes:
        return (ROOT / self.path).read_bytes()

    def text(self) -> str:
        b = self.bytes()
        return b.decode("utf-8", errors="replace")


def _throttle(host: str) -> None:
    gap = MIN_INTERVAL.get(host, 1.0)
    t = _last_hit.get(host, 0)
    wait = t + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_hit[host] = time.time()


def log_access(url: str, outcome: str, status: int | None, detail: str = "") -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACCESS_LOG.open("a") as f:
        f.write(json.dumps({"url": url, "outcome": outcome, "status": status,
                            "detail": detail[:300], "at": now_iso()}) + "\n")


def _is_challenge(resp: requests.Response) -> bool:
    if resp.status_code in (403, 503, 429):
        body = resp.text[:3000].lower()
        return ("just a moment" in body or "cf-chl" in body or "challenge" in body
                or "attention required" in body or resp.headers.get("server", "").lower() == "cloudflare")
    return False


def _get(url: str, timeout: int = 40) -> requests.Response:
    host = urllib.parse.urlparse(url).netloc
    _throttle(host)
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en-CA,en;q=0.8"},
                             timeout=timeout, allow_redirects=True)
            if r.status_code in (429, 502, 503, 504) and "archive.org" in host:
                time.sleep(10 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    return r


def _ext(content_type: str, url: str) -> str:
    ct = content_type.lower()
    if "pdf" in ct or url.lower().endswith(".pdf"):
        return "pdf"
    if "json" in ct:
        return "json"
    if "csv" in ct:
        return "csv"
    return "html"


def _store(url: str, resp: requests.Response, via: str, wayback_ts: str | None = None,
           note: str | None = None) -> Fetched:
    data = resp.content
    if data[:2] == b"\x1f\x8b":  # wayback id_ sometimes returns raw gzip
        try:
            data = gzip.decompress(data)
        except OSError:
            pass
    key = url_key(url)
    d = CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    ext = _ext(resp.headers.get("content-type", ""), url)
    p = d / f"raw.{ext}"
    p.write_bytes(data)
    f = Fetched(url=url, final_url=resp.url, via=via, status=resp.status_code,
                content_type=resp.headers.get("content-type", ""),
                sha256=hashlib.sha256(data).hexdigest(), retrieved_at=now_iso(),
                path=str(p.relative_to(ROOT)), wayback_ts=wayback_ts, note=note)
    (d / "meta.json").write_text(json.dumps(asdict(f), indent=1))
    return f


def cached(url: str) -> Fetched | None:
    m = CACHE / url_key(url) / "meta.json"
    if m.exists():
        return Fetched(**json.loads(m.read_text()))
    return None


def wayback_latest(url: str) -> str | None:
    """Most recent 200 snapshot timestamp for url via the CDX API."""
    q = ("https://web.archive.org/cdx/search/cdx?output=json&fl=timestamp,statuscode"
         f"&filter=statuscode:200&url={urllib.parse.quote(url, safe='')}")
    try:
        r = _get(q)
        rows = r.json() if r.status_code == 200 and r.text.strip().startswith("[") else []
    except (requests.RequestException, ValueError):
        return None
    rows = rows[1:]
    return rows[-1][0] if rows else None


def fetch(url: str, *, allow_wayback: bool = True, refresh: bool = False,
          wayback_ts: str | None = None) -> Fetched | None:
    """Fetch url (cached). Falls back to the Internet Archive when the origin
    refuses automated access. Returns None and logs when unavailable.
    A page captured by a person's browser (captures.py) takes precedence."""
    from . import captures
    cap = captures.lookup(url)
    if cap:
        return Fetched(url=url, final_url=cap["url"], via="user_browser_capture", status=200,
                       content_type="application/pdf" if cap["kind"] == "pdf" else "text/html",
                       sha256=cap["sha256"], retrieved_at=cap["captured_at"], path=cap["path"],
                       note=f"Captured in a user's browser at {cap['captured_at']}")
    if not refresh:
        c = cached(url)
        if c:
            return c
    if wayback_ts:
        return _fetch_wayback(url, wayback_ts)
    try:
        r = _get(url)
    except requests.RequestException as e:
        log_access(url, "network_error", None, repr(e))
        r = None
    if r is not None and r.status_code == 200 and not _is_challenge(r):
        log_access(url, "ok_direct", 200)
        return _store(url, r, "direct")
    if r is not None:
        outcome = "blocked_bot_challenge" if _is_challenge(r) else f"http_{r.status_code}"
        log_access(url, outcome, r.status_code)
    if not allow_wayback:
        return None
    ts = wayback_latest(url)
    if not ts:
        log_access(url, "no_archive_snapshot", None)
        return None
    return _fetch_wayback(url, ts)


def _fetch_wayback(url: str, ts: str) -> Fetched | None:
    wb = f"https://web.archive.org/web/{ts}id_/{url}"
    try:
        r = _get(wb, timeout=60)
    except requests.RequestException as e:
        log_access(url, "wayback_error", None, repr(e))
        return None
    if r.status_code != 200:
        log_access(url, f"wayback_http_{r.status_code}", r.status_code)
        return None
    log_access(url, "ok_wayback", 200, ts)
    return _store(url, r, "wayback", wayback_ts=ts,
                  note=f"Origin refused automated access; Internet Archive snapshot {ts}")
