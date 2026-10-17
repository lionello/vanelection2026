"""Official corroboration from Elections BC's registered-candidates list
(Local Elections Campaign Financing, 2026 General Local Elections).

Only jurisdiction, office, affiliation and candidate name are read. Financial
agents' names and addresses in the same PDF are deliberately ignored.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import pymupdf

from .fetch import ROOT

PDF = ROOT / "data" / "official" / "elections-bc-registered-candidates-LEGE-2026.pdf"
URL = "https://elections.bc.ca/docs/lecfa/Registered-Candidates-LEGE-2026-10-17.pdf"
COLS = {"jur": (0, 120), "office": (120, 215), "name": (215, 355), "aff": (355, 460)}


def rows():
    doc = pymupdf.open(PDF)
    jur = None
    for pno, page in enumerate(doc, start=1):
        lines = []
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                s = " ".join(sp["text"] for sp in l["spans"]).strip()
                if s:
                    lines.append((round(l["bbox"][1]), l["bbox"][0], s))
        by_y = defaultdict(dict)
        for y, x, s in lines:
            if y <= 25:
                continue  # header
            for col, (a, b) in COLS.items():
                if a <= x < b:
                    # merge rows within 2pt
                    yk = next((k for k in by_y if abs(k - y) <= 2), y)
                    by_y[yk][col] = s
        for y in sorted(by_y):
            r = by_y[y]
            if "jur" in r:
                # a jurisdiction line without an office is the wrapped tail of the previous label
                cont = r["jur"].split()[0] in ("District", "School", "City", "of", "No.") and jur
                jur = f"{jur} {r['jur']}" if ("office" not in r or cont) and jur else r["jur"]
            if "office" in r and "name" in r:
                yield dict(jurisdiction=jur, office=r["office"], name=r["name"], affiliation=r.get("aff"), page=pno)


def vancouver() -> list[dict]:
    return [r for r in rows() if r["jurisdiction"] in ("Vancouver", "Vancouver School District")]


def sha() -> str:
    return hashlib.sha256(PDF.read_bytes()).hexdigest()


if __name__ == "__main__":
    from collections import Counter
    v = vancouver()
    print(len(v), Counter((r["jurisdiction"], r["office"]) for r in v))
    jurs = Counter(r["jurisdiction"] for r in rows() if r["jurisdiction"] and "ancouver" in r["jurisdiction"])
    print(jurs)
