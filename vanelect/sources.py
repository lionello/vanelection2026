"""Source registry, fetching and passage extraction.

Inputs:  data/seeds/*.json (discovered URLs, see AGENT_BRIEF.md) + CORE sources.
Outputs: data/processed/sources.json   one record per source (provenance, hashes)
         data/processed/passages.jsonl one record per passage (page / block ids)
         data/text/<source_id>.txt     extracted text, for inspection
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import fitz  # pymupdf
import trafilatura
from bs4 import BeautifulSoup

from .fetch import ROOT, fetch, cached, now_iso, Fetched

SEEDS = ROOT / "data" / "seeds"
OUT = ROOT / "data" / "processed"
TEXT = ROOT / "data" / "text"
SNAP = ROOT / "data" / "official_snapshots"

SOURCE_TYPES = ["candidate_bio", "campaign_site", "party_platform", "party_candidate_list",
                "official_record", "employer_bio", "news", "questionnaire", "advocacy_assessment",
                "commentary", "reference"]

# Sources stored locally at project start (fetched before the fetch cache existed).
CORE = [
    dict(url="https://vancouver.ca/your-government/2026-candidates-mayor.aspx", title="2026 mayoral candidates",
         publisher="City of Vancouver", published=None, source_type="official_record", candidates=["*mayor"],
         local="2026-candidates-mayor.wb20261012082952.html", wayback_ts="20261012082952",
         notes="Official final candidate list (archived)."),
    dict(url="https://vancouver.ca/your-government/2026-candidates-park-board-commissioner.aspx",
         title="2026 Park Board commissioner candidates", publisher="City of Vancouver", published=None,
         source_type="official_record", candidates=["*park"],
         local="2026-candidates-park-board-commissioner.wb20260930073937.html", wayback_ts="20260930073937",
         notes="Official final candidate list (archived)."),
    dict(url="https://vancouver.ca/your-government/2026-candidates-school-trustee.aspx",
         title="2026 school trustee candidates", publisher="City of Vancouver", published=None,
         source_type="official_record", candidates=["*school"],
         local="2026-candidates-school-trustee.wb20261001164108.html", wayback_ts="20261001164108",
         notes="Official final candidate list (archived)."),
    dict(url="https://dailyhive.com/vancouver/vancouver-2026-civic-election-candidates-ballot-order",
         title="Vancouver's 2026 civic election candidate list is set: All 130 names in ballot order",
         publisher="Daily Hive", published="2026-09-22", source_type="news", candidates=["*council"],
         local="dailyhive-ballot-order-2026-09-22.html", notes="Ballot order list; council list source."),
    dict(url="https://allenpike.com/2026/2026-vancouver-election-guide/", title="Vancouver Election Guide, 2026",
         publisher="Allen Pike (personal blog)", published="2026-10-12", source_type="commentary", candidates=[],
         local="allenpike-2026-guide.html",
         perspective="Personal guide by a Vancouver tech executive; self-described urbanist framing; "
                     "makes explicit voting recommendations (shown here as commentary only).",
         notes="Secondary context; never used as factual verification."),
]


def source_id(url: str) -> str:
    return "S" + hashlib.sha1(url.strip().encode()).hexdigest()[:10]


def load_seeds() -> tuple[list[dict], list[dict], list[dict]]:
    sources, affs, gaps = [], [], []
    for p in sorted(SEEDS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"!! seed {p.name} unreadable: {e}")
            continue
        for s in d.get("sources", []):
            s["seed_file"] = p.name
            sources.append(s)
        for a in d.get("affiliations", []):
            a["seed_file"] = p.name
            affs.append(a)
        for g in d.get("gaps", []):
            g["seed_file"] = p.name
            gaps.append(g)
    return sources, affs, gaps


def merge_sources(seed_sources: list[dict], valid_ids: set[str]) -> list[dict]:
    """One registry entry per URL; candidate lists unioned across seed files."""
    reg: dict[str, dict] = {}
    for s in CORE + seed_sources:
        url = s["url"].strip()
        url = re.sub(r"[?&]utm_[^&#]+", "", url)
        sid = source_id(url)
        cands = [c for c in s.get("candidates", []) if c in valid_ids or c.startswith("*")]
        bad = [c for c in s.get("candidates", []) if c not in valid_ids and not c.startswith("*")]
        if sid in reg:
            r = reg[sid]
            r["candidates"] = sorted(set(r["candidates"]) | set(cands))
            r["topics"] = sorted(set(r.get("topics") or []) | set(s.get("topics") or []))
            r["seed_files"] = sorted(set(r["seed_files"]) | {s.get("seed_file", "core")})
            continue
        st = s.get("source_type") or "reference"
        if st not in SOURCE_TYPES:
            st = "reference"
        reg[sid] = dict(source_id=sid, url=url, title=s.get("title"), publisher=s.get("publisher"),
                        published=s.get("published"), source_type=st, candidates=sorted(set(cands)),
                        unknown_candidate_ids=bad, party=s.get("party"), topics=s.get("topics") or [],
                        perspective=s.get("perspective"), access_hint=s.get("access"),
                        wayback_ts=s.get("wayback_ts"), notes=s.get("notes"), local=s.get("local"),
                        seed_files=[s.get("seed_file", "core")])
    return list(reg.values())


# ----------------------------------------------------------------------------- extraction
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "td", "th", "dd", "dt", "figcaption"]
DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "aside", "svg", "iframe", "button"]


def clean(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


def html_blocks(raw: str) -> list[dict]:
    soup = BeautifulSoup(raw, "lxml")
    for t in soup(DROP_TAGS):
        t.decompose()
    for sel in ["[class*=cookie]", "[id*=cookie]", "[class*=newsletter]", "[class*=share]", "[class*=related]",
                "[class*=breadcrumb]", "[class*=menu]", "[role=navigation]"]:
        for t in soup.select(sel):
            t.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    blocks, seen = [], set()
    for el in main.find_all(BLOCK_TAGS):
        if el.find(BLOCK_TAGS):  # only leaf-ish blocks
            if el.name not in ("li", "td") or el.find(["p", "li"]):
                continue
        txt = clean(el.get_text(" "))
        if len(txt) < 3 or txt in seen:
            continue
        seen.add(txt)
        blocks.append(dict(kind="heading" if el.name[0] == "h" and el.name[1:].isdigit() else "text",
                           level=int(el.name[1]) if el.name[0] == "h" and el.name[1:].isdigit() else None,
                           text=txt))
    body_chars = sum(len(b["text"]) for b in blocks if b["kind"] == "text")
    if body_chars < 400:  # JS-heavy page: fall back to trafilatura's main-text heuristic
        t = trafilatura.extract(raw, output_format="markdown", include_formatting=False) or ""
        blocks = []
        for para in re.split(r"\n\s*\n", t):
            para = para.strip()
            if not para:
                continue
            if para.startswith("#"):
                blocks.append(dict(kind="heading", level=len(para) - len(para.lstrip("#")), text=clean(para.lstrip("# "))))
            else:
                blocks.append(dict(kind="text", level=None, text=clean(para)))
    return blocks


def pdf_blocks(path: Path) -> list[dict]:
    doc = fitz.open(path)
    blocks = []
    for pno, page in enumerate(doc, start=1):
        for b in page.get_text("blocks"):
            txt = clean(re.sub(r"(\w)-\n([a-z])", r"\1\2", b[4]))  # undo line-break hyphenation
            if len(txt) >= 3:
                blocks.append(dict(kind="text", level=None, text=txt, page=pno))
    return blocks


def extract(src: dict) -> tuple[list[dict], dict]:
    """Return (passages, fetch_meta) for a registry source."""
    if src.get("local"):
        p = SNAP / src["local"]
        data = p.read_bytes()
        fmeta = dict(via="local_snapshot" if not src.get("wayback_ts") else "wayback",
                     wayback_ts=src.get("wayback_ts"), status=200, sha256=hashlib.sha256(data).hexdigest(),
                     retrieved_at="2026-10-17T11:58:00+00:00", path=str(p.relative_to(ROOT)), content_type="text/html")
        blocks = html_blocks(data.decode("utf-8", errors="replace"))
    else:
        f = fetch(src["url"], wayback_ts=src.get("wayback_ts") if src.get("access_hint") == "blocked_use_wayback" else None)
        if f is None:
            return [], dict(status=None, via="unavailable")
        fmeta = dict(via=f.via, wayback_ts=f.wayback_ts, status=f.status, sha256=f.sha256,
                     retrieved_at=f.retrieved_at, path=f.path, content_type=f.content_type, final_url=f.final_url)
        p = ROOT / f.path
        if p.suffix == ".pdf":
            try:
                blocks = pdf_blocks(p)
            except Exception as e:  # corrupt / scanned
                fmeta["extract_error"] = repr(e)
                blocks = []
        else:
            blocks = html_blocks(p.read_bytes().decode("utf-8", errors="replace"))
    passages, heading = [], []
    for i, b in enumerate(blocks):
        if b["kind"] == "heading":
            lvl = b["level"] or 2
            heading = [h for h in heading if h[0] < lvl] + [(lvl, b["text"])]
        pid = f"{src['source_id']}:{'p' + str(b['page']) + ':' if b.get('page') else ''}b{i}"
        passages.append(dict(passage_id=pid, source_id=src["source_id"], page=b.get("page"), block=i,
                             kind=b["kind"], level=b.get("level"), heading_path=[h[1] for h in heading] if b["kind"] == "text" else
                             [h[1] for h in heading[:-1]], text=b["text"]))
    return passages, fmeta


def build(refetch: bool = False) -> dict:
    cands = json.loads((OUT / "candidates.json").read_text())
    valid = {c["candidate_id"] for c in cands}
    seeds, affs, gaps = load_seeds()
    reg = merge_sources(seeds, valid)
    TEXT.mkdir(parents=True, exist_ok=True)
    all_passages, stats = [], {"ok": 0, "unavailable": 0, "empty": 0}
    for i, s in enumerate(reg):
        passages, fmeta = extract(s)
        s["fetch"] = fmeta
        s["content_hash"] = fmeta.get("sha256")
        s["n_passages"] = sum(p["kind"] == "text" for p in passages)
        s["n_chars"] = sum(len(p["text"]) for p in passages if p["kind"] == "text")
        if fmeta.get("status") is None:
            s["availability"] = "inaccessible"
            stats["unavailable"] += 1
        elif s["n_chars"] < 200:
            s["availability"] = "fetched_but_no_text"
            stats["empty"] += 1
        else:
            s["availability"] = "ok"
            stats["ok"] += 1
        (TEXT / f"{s['source_id']}.txt").write_text(
            f"# {s['title']}\n# {s['url']}\n# via={fmeta.get('via')} ts={fmeta.get('wayback_ts')} sha256={fmeta.get('sha256')}\n\n"
            + "\n\n".join(("## " if p["kind"] == "heading" else f"[{p['passage_id']}] ") + p["text"] for p in passages))
        all_passages.extend(passages)
        if (i + 1) % 25 == 0:
            print(f"  sources {i + 1}/{len(reg)} {stats}", flush=True)
    # syndicated / duplicate documents: identical or near-identical text across URLs
    mark_duplicate_sources(reg, all_passages)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False))
    with (OUT / "passages.jsonl").open("w") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (OUT / "affiliations_seed.json").write_text(json.dumps(affs, indent=1, ensure_ascii=False))
    (OUT / "gaps_seed.json").write_text(json.dumps(gaps, indent=1, ensure_ascii=False))
    return dict(sources=len(reg), passages=len(all_passages), **stats, built_at=now_iso())


def shingles(text: str, k: int = 5) -> set[int]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(" ".join(w[i:i + k])) for i in range(max(0, len(w) - k + 1))}


def mark_duplicate_sources(reg: list[dict], passages: list[dict], thr: float = 0.8) -> None:
    by = {}
    for p in passages:
        if p["kind"] == "text":
            by.setdefault(p["source_id"], []).append(p["text"])
    sh = {sid: shingles(" ".join(t)) for sid, t in by.items()}
    ids = [s["source_id"] for s in reg if sh.get(s["source_id"])]
    for s in reg:
        s["duplicate_of"] = None
    order = sorted(ids, key=lambda i: next((x.get("published") or "9999") for x in reg if x["source_id"] == i))
    for i, a in enumerate(order):
        sa = sh[a]
        for b in order[:i]:
            sb = sh[b]
            if not sa or not sb:
                continue
            j = len(sa & sb) / min(len(sa), len(sb))
            if j >= thr:
                ra = next(x for x in reg if x["source_id"] == a)
                if ra["duplicate_of"] is None and next(x for x in reg if x["source_id"] == b)["duplicate_of"] is None:
                    ra["duplicate_of"] = b
                    ra["duplicate_containment"] = round(j, 3)
                break


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
