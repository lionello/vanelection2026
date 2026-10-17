"""Recorded votes parsed from Park Board / School Board minutes PDFs.

Only minutes that are in the source registry (official_record, cached PDF)
are parsed, so coverage is partial and reported as such. Each motion keeps
the minutes' own wording for the result and member lists:

  CARRIED / CARRIED UNANIMOUSLY / LOST / DEFEATED / WITHDRAWN / REFERRED
  In favour: Commissioners A, B   Opposed: ...   Absent: ...   Abstained: ...
  "declared a conflict of interest" -> recusal

Vote values stay distinct: in_favour, opposed, absent, abstained, recused.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import pymupdf

from .fetch import ROOT

OUT = ROOT / "data" / "processed"
RESULT = re.compile(r"^\s*(CARRIED UNANIMOUSLY|CARRIED|LOST|DEFEATED|WITHDRAWN|REFERRED|TABLED)\b(.*)$", re.M)
LISTS = {
    "in_favour": re.compile(r"In favou?r\s*:\s*(.+?)(?=\n\s*(?:Opposed|Against|Absent|Abstain|Not present|Recused)|\n\s*\n|$)", re.I | re.S),
    "opposed": re.compile(r"(?:Opposed|Against)\s*:\s*(.+?)(?=\n\s*(?:In favou?r|Absent|Abstain|Not present|Recused)|\n\s*\n|$)", re.I | re.S),
    "absent": re.compile(r"(?:Absent|Not present)(?: for (?:the )?vote)?\s*:\s*(.+?)(?=\n\s*(?:In favou?r|Opposed|Abstain|Recused)|\n\s*\n|$)", re.I | re.S),
    "abstained": re.compile(r"Abstain(?:ed|ing)?\s*:\s*(.+?)(?=\n\s*(?:In favou?r|Opposed|Absent|Recused)|\n\s*\n|$)", re.I | re.S),
}
PAREN_OPPOSED = re.compile(r"\((?:Commissioners?|Trustees?)\s+(.+?)\s+opposed\)", re.I | re.S)
CONFLICT = re.compile(r"(?:Commissioner|Trustee)\s+([A-Z][\w'\-]+)\s+declared\s+(?:a\s+)?(?:conflict|an interest)", re.I)
MOVED = re.compile(r"MOVED by (?:Commissioner|Trustee)\s+([A-Z][\w'\-]+)", re.I)
DATE = re.compile(r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{1,2}),\s+(20\d\d)", re.I)
MONTHS = {m: i for i, m in enumerate(["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
                                       "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"], 1)}

# surname (as written in minutes) -> candidate_id, for candidates who sat on these boards
PARK_MEMBERS = {"Digby": "park-tom-digby", "Jensen": "park-scott-jensen", "Virdi": "park-jas-virdi",
                "Barker": "park-tricia-barker", "Irwin": "park-john-irwin", "Dumont": "council-camil-dumont",
                "Giesbrecht": "school-gwen-giesbrecht", "Coupar": "council-john-coupar", "Wyss": None}
SCHOOL_MEMBERS = {"Mah": "school-suzie-mah", "Chien": "school-alfred-chien", "Faridkot": "school-preeti-faridkot",
                  "Richardson": "school-christopher-jk-richardson", "Jung": "council-victoria-jung"}


def names_in(s: str) -> list[str]:
    s = re.sub(r"\b(Commissioners?|Trustees?|Chair|Vice-Chair|and|Nil|None)\b", ",", s)
    out = []
    for tok in re.split(r"[,;\n]+", s):
        tok = tok.strip(" .*")
        if re.fullmatch(r"[A-Z][\w'\-]+(?: [A-Z][\w'\-]+)?", tok):
            out.append(tok.split()[-1])
    return out


def parse(text: str) -> list[dict]:
    motions = []
    prev = 0
    for m in RESULT.finditer(text):
        block = text[prev:m.start()]
        tail = text[m.end(): m.end() + 600]
        cut = [x.start() for x in (MOVED.search(tail), RESULT.search(tail)) if x]
        if cut:
            tail = tail[:min(cut)]  # lists belong to this result only, never the next motion
        prev = m.end()
        mv = list(MOVED.finditer(block))
        if not mv:
            continue
        body = block[mv[-1].start():]
        that = re.search(r"\b(THAT|A\.|B\.)\b(.{0,400})", body, re.S)
        qual = re.match(r"\s*([A-Z0-9/ ,]{4,})\n", tail)  # e.g. "NOT HAVING RECEIVED / THE REQUIRED 2/3 MAJORITY"
        result = (m.group(1) + " " + m.group(2).strip() + (" " + qual.group(1).strip() if qual and re.search(r"NOT HAVING|REQUIRED|MAJORITY|AMENDED", qual.group(1)) else "")).strip()
        rec = dict(result=re.sub(r"\s+", " ", result).upper(), moved_by=mv[-1].group(1), text=re.sub(r"\s+", " ", (that.group(0) if that else body))[:400],
                   members=defaultdict(str))
        lists_txt = tail.split("\n\n\n")[0]
        for k, rx in LISTS.items():
            x = rx.search(lists_txt)
            if x:
                for n in names_in(x.group(1)):
                    rec["members"][n] = k
        po = PAREN_OPPOSED.search(m.group(2) + tail[:200])
        if po:
            for n in names_in(po.group(1)):
                rec["members"][n] = "opposed"
        for c in CONFLICT.finditer(block):
            rec["members"][c.group(1)] = "recused"
        rec["is_amendment"] = bool(re.search(r"\bAMENDMENT\b|\bamend(ed)?\b", body[:300], re.I))
        rec["members"] = dict(rec["members"])
        # sanity: a unanimous result cannot carry named opposition
        if rec["result"].startswith("CARRIED UNANIMOUSLY") and "opposed" in rec["members"].values():
            rec["parse_warning"] = "unanimous result with named opposition - check minutes"
        motions.append(rec)
    return motions


def build() -> dict:
    sources = json.loads((OUT / "sources.json").read_text())
    out, per = [], defaultdict(lambda: defaultdict(int))
    meetings = []
    for s in sources:
        path = (s.get("fetch") or {}).get("path")
        if s["source_type"] != "official_record" or not path or not path.endswith(".pdf"):
            continue
        url = s["url"].lower()
        board = "park" if "parkboard" in url else "school" if "vsb" in url else None
        if not board:
            continue
        try:
            doc = pymupdf.open(ROOT / path)
            text = "\n".join(p.get_text() for p in doc)
        except Exception:
            continue
        if "MINUTES" not in text[:3000].upper():
            continue
        d = DATE.search(text[:2000])
        date = f"{d.group(3)}-{MONTHS[d.group(1).upper()]:02d}-{int(d.group(2)):02d}" if d else None
        ms = parse(text)
        members = PARK_MEMBERS if board == "park" else SCHOOL_MEMBERS
        meetings.append(dict(source_id=s["source_id"], url=s["url"], board=board, date=date, motions=len(ms),
                             with_member_lists=sum(1 for m in ms if m["members"])))
        for i, m in enumerate(ms):
            m.update(board=board, date=date, source_id=s["source_id"], motion_id=f"{s['source_id']}:{i}")
            m["candidate_votes"] = {members[n]: v for n, v in m["members"].items() if members.get(n)}
            out.append(m)
            for cid, v in m["candidate_votes"].items():
                per[cid][v] += 1
    res = dict(meetings=meetings, motions=out, per_candidate={k: dict(v) for k, v in per.items()},
               note=("Parsed from the Park Board / School Board minutes present in the source registry only. "
                     "Partial coverage; unanimous motions often list no names. Values: in_favour, opposed, absent, "
                     "abstained, recused - only in_favour/opposed are votes."))
    (OUT / "board_votes.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return dict(meetings=len(meetings), motions=len(out), with_lists=sum(1 for m in out if m["members"]),
                per_candidate=res["per_candidate"])


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
