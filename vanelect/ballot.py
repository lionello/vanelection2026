"""Build the verified ballot (candidates.json) from official City of Vancouver pages.

Provenance policy
-----------------
* vancouver.ca sits behind a Cloudflare bot challenge that refuses automated
  clients (HTTP 403). We do not try to defeat it. The official candidate pages
  are read from public Internet Archive snapshots of vancouver.ca, which are
  byte-for-byte captures of the City's own HTML.
* The councillor page (2026-candidates-councillor.aspx) has no successful
  archive capture (the only one is itself a 403 challenge), and the City's
  candidate-list PDF was never archived. Council candidates therefore come from
  Daily Hive's ballot-order article (2026-09-22), which reproduces the City's
  list. Every entry in every race is then matched against Elections BC's
  official registered-candidates list (electionsbc.py); council entries become
  `official_registration_plus_secondary_ballot_number`: name/office official,
  ballot number from the secondary source.
  Daily Hive's mayor / Park Board / school lists are cross-checked against the
  official pages; agreement is recorded in ballot_meta.json.
"""
from __future__ import annotations

import html as htmlmod
import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup

from .fetch import ROOT, now_iso

SNAP = ROOT / "data" / "official_snapshots"
OUT = ROOT / "data" / "processed"

RACES = {
    "mayor": dict(label="Mayor", seats=1, page="2026-candidates-mayor.aspx",
                  snapshot="2026-candidates-mayor.wb20261012082952.html", ts="20261012082952"),
    "council": dict(label="City Council", seats=10, page="2026-candidates-councillor.aspx",
                    snapshot=None, ts=None),
    "park": dict(label="Park Board", seats=7, page="2026-candidates-park-board-commissioner.aspx",
                 snapshot="2026-candidates-park-board-commissioner.wb20260930073937.html",
                 ts="20260930073937"),
    "school": dict(label="School Board", seats=9, page="2026-candidates-school-trustee.aspx",
                   snapshot="2026-candidates-school-trustee.wb20261001164108.html",
                   ts="20261001164108"),
}
OFFICIAL_BASE = "https://vancouver.ca/your-government/"
DAILYHIVE_URL = "https://dailyhive.com/vancouver/vancouver-2026-civic-election-candidates-ballot-order"

# Canonical party names. Ballot labels are kept verbatim alongside.
PARTY_CANON = {
    "ABC Vancouver": "ABC Vancouver",
    "TEAM": "TEAM for a Livable Vancouver",
    "Green": "Green Party of Vancouver",
    "Vote Vancouver": "Vote Vancouver",
    "Bright Future Vancouver": "Bright Future Vancouver",
    "Vancouver Liberals": "Vancouver Liberals",
    "COPE": "COPE",
    "OneCity": "OneCity Vancouver",
    "Independent Parks Party": "Independent Parks Party",
    "TrueBlue Vancouver": "TrueBlue Vancouver",
    "Affordable Housing": "Affordable Housing",
}
PARTY_ID = {
    "ABC Vancouver": "abc", "TEAM for a Livable Vancouver": "team",
    "Green Party of Vancouver": "green", "Vote Vancouver": "votevan",
    "Bright Future Vancouver": "bfv", "Vancouver Liberals": "vanlib", "COPE": "cope",
    "OneCity Vancouver": "onecity", "Independent Parks Party": "ipp",
    "TrueBlue Vancouver": "trueblue", "Affordable Housing": "affordablehousing",
}


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s


def strip_cjk(name: str) -> tuple[str, str | None]:
    """Split 'Chris 仇云霞' into ('Chris', '仇云霞')."""
    m = re.search(r"[　-鿿豈-﫿]+", name)
    if not m:
        return name.strip(), None
    return (name[:m.start()] + name[m.end():]).strip(), m.group(0)


def candidate_id(race: str, display_name: str) -> str:
    return f"{race}-{slug(display_name)}"


def official_page(race: str) -> tuple[Path | None, dict | None]:
    """Prefer a browser capture of the City's page; else the archived snapshot."""
    from . import captures
    cap = captures.lookup(OFFICIAL_BASE + RACES[race]["page"])
    if cap and cap["kind"] == "html":
        return ROOT / cap["path"], cap
    snap = RACES[race]["snapshot"]
    return (SNAP / snap if snap else None), None


def parse_official(race: str, path: Path | None = None) -> list[dict]:
    info = RACES[race]
    raw = (path or SNAP / info["snapshot"]).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")
    out = []
    for li in soup.select("li"):
        summ = li.select_one(".pd-blk-summary-content")
        if not summ:
            continue
        num = summ.find("strong")
        h2 = summ.find("h2")
        if not num or not h2 or not num.get_text(strip=True).isdigit():
            continue
        body = li.select_one("[id^=pd-blk-group]")
        ballot_name = htmlmod.unescape(h2.get_text(" ", strip=True)).replace("\xa0", " ").strip()
        party_label = None
        links, docs = [], []
        if body:
            for p in body.find_all("p"):
                t = p.get_text(" ", strip=True).replace("\xa0", " ")
                if t.startswith("Party:"):
                    party_label = t.split(":", 1)[1].strip() or None
            for a in body.find_all("a"):
                href = a.get("href", "")
                label = a.get_text(" ", strip=True).replace("External website, opens in new tab", "").strip()
                if href.startswith("/cdn-cgi/") or href.startswith("tel:"):
                    continue  # obfuscated e-mail / phone: contact data, not evidence
                if href.startswith("/files/"):
                    docs.append({"label": label.split(" PDF")[0], "url": "https://vancouver.ca" + href})
                elif href.startswith("http"):
                    links.append({"label": label, "url": href})
        last, _, first = ballot_name.partition(",")
        first, cjk = strip_cjk(first)
        display = f"{first.strip()} {last.strip().title() if last.isupper() else last.strip()}".strip()
        out.append(dict(ballot_number=int(num.get_text(strip=True)), ballot_name=ballot_name,
                        display_name=fix_case(display, last), name_cjk=cjk, party_label=party_label,
                        campaign_links=links, official_documents=docs))
    return out


def fix_case(display: str, last_upper: str) -> str:
    # Ballot surnames are upper-case; restore conventional casing for display.
    last = last_upper.strip()
    nice = " ".join(w.capitalize() if not w.startswith("O'") else "O'" + w[2:].capitalize()
                    for w in last.lower().split())
    nice = re.sub(r"\bMc(\w)", lambda m: "Mc" + m.group(1).upper(), nice)
    nice = nice.replace("Van Vloten", "Van Vloten").replace("O'keefe", "O'Keefe")
    first = display[: len(display) - len(last.title())].strip() if display.lower().endswith(last.lower()) else display
    return f"{first} {nice}".strip()


def parse_dailyhive() -> dict[str, list[dict]]:
    raw = (SNAP / "dailyhive-ballot-order-2026-09-22.html").read_text(encoding="utf-8", errors="replace")
    text = BeautifulSoup(raw, "lxml").get_text("\n")
    text = htmlmod.unescape(text).replace("’", "'")
    sections = {"mayor": r"candidates for Vancouver mayor", "council": r"candidates for Vancouver City Council",
                "park": r"candidates for Vancouver Park Board", "school": r"candidates for Vancouver School Board"}
    idx = {k: text.find(re.search(v, text).group(0)) for k, v in sections.items()}
    order = sorted(idx, key=idx.get)
    res = {}
    for i, k in enumerate(order):
        seg = text[idx[k]: idx[order[i + 1]] if i + 1 < len(order) else None]
        rows = []
        for m in re.finditer(r"^\s*(\d{2,3}):\s*(.+?)\s*$", seg, re.M):
            entry = m.group(2)
            pm = re.match(r"(.*?)\s*\(([^)]+)\)\s*$", entry)
            name, party = (pm.group(1), pm.group(2)) if pm else (entry, None)
            rows.append(dict(ballot_number=int(m.group(1)), name=name.strip(), party_label=party))
        res[k] = rows
    return res


def norm_name(s: str) -> str:
    s, _ = strip_cjk(s)
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())


NAME_VARIANTS = {("council", "mikeklassen"): "michaelklassen", ("school", "traversfenton"): "fentontravers"}


def electionsbc_check(candidates: list[dict]) -> dict:
    """Corroborate every ballot entry against Elections BC's official registered-candidates
    list (campaign-financing registration). Mutates candidates: adds official_registration."""
    from . import electionsbc
    off = {"Councillor": "council", "Park Board": "park", "Board of Education": "school", "Mayor": "mayor"}
    eb = {}
    for r in electionsbc.rows():
        if r["jurisdiction"] in ("Vancouver", "Vancouver School District", "Vancouver School") and r["office"] in off:
            eb[(off[r["office"]], norm_name(r["name"]))] = r
    matched, unmatched = 0, []
    for c in candidates:
        k = (c["race"], norm_name(c["display_name"]))
        how = "exact_name"
        if k not in eb and k in NAME_VARIANTS:
            k, how = (c["race"], NAME_VARIANTS[k]), "name_variant"
        r = eb.pop(k, None)
        if r:
            matched += 1
            c["official_registration"] = dict(source="Elections BC", url=electionsbc.URL, page=r["page"],
                                              name_as_registered=r["name"], affiliation_registered=r["affiliation"],
                                              office_registered=r["office"], match=how)
            if c["ballot_verification"] == "secondary_source":
                c["ballot_verification"] = "official_registration_plus_secondary_ballot_number"
        else:
            c["official_registration"] = None
            unmatched.append(c["candidate_id"])
    return dict(url=electionsbc.URL, sha256=electionsbc.sha(), matched=matched, unmatched_ballot=unmatched,
                registered_not_on_ballot=[dict(name=r["name"], office=r["office"], affiliation=r["affiliation"])
                                          for r in eb.values()],
                note=("Elections BC lists candidates registered for campaign-financing purposes; it corroborates "
                      "identity, office and affiliation but not ballot numbers. Financial-agent details in the "
                      "same document are not read."))


def ensure_inputs() -> None:
    """Re-download the official inputs if this is a fresh checkout without them."""
    import requests
    from . import electionsbc
    SNAP.mkdir(parents=True, exist_ok=True)
    for race, info in RACES.items():
        if info["snapshot"] and not (SNAP / info["snapshot"]).exists():
            r = requests.get(f"https://web.archive.org/web/{info['ts']}id_/{OFFICIAL_BASE}{info['page']}", timeout=60)
            r.raise_for_status()
            (SNAP / info["snapshot"]).write_bytes(r.content)
    dh = SNAP / "dailyhive-ballot-order-2026-09-22.html"
    if not dh.exists():
        dh.write_bytes(requests.get(DAILYHIVE_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).content)
    if not electionsbc.PDF.exists():
        electionsbc.PDF.parent.mkdir(parents=True, exist_ok=True)
        electionsbc.PDF.write_bytes(requests.get(electionsbc.URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"}).content)


def build() -> dict:
    ensure_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    dh = parse_dailyhive()
    candidates, checks = [], {}
    for race, info in RACES.items():
        page, cap = official_page(race)
        rows = parse_official(race, page) if page else []
        if page and rows:
            if cap:
                src = dict(kind="official_user_capture", url=OFFICIAL_BASE + info["page"],
                           captured_at=cap["captured_at"], sha256=cap["sha256"],
                           note="Official City page captured in a person's browser (site refuses automated access).")
            else:
                src = dict(kind="official_archived", url=OFFICIAL_BASE + info["page"],
                           archive_url=f"https://web.archive.org/web/{info['ts']}/{OFFICIAL_BASE}{info['page']}",
                           snapshot_ts=info["ts"])
            # cross-check with Daily Hive
            dmap = {r["ballot_number"]: r for r in dh[race]}
            mism = []
            for r in rows:
                d = dmap.get(r["ballot_number"])
                first_last = r["display_name"]
                if not d or norm_name(d["name"]) != norm_name(first_last):
                    mism.append({"ballot": r["ballot_number"], "official": first_last, "dailyhive": d and d["name"]})
            checks[race] = dict(official_count=len(rows), dailyhive_count=len(dh[race]), name_mismatches=mism,
                                official_source=src["kind"])
            verification = "official_user_capture" if cap else "official_archived_snapshot"
        else:
            rows = [dict(ballot_number=r["ballot_number"], ballot_name=None,
                         display_name=strip_cjk(r["name"])[0], name_cjk=strip_cjk(r["name"])[1],
                         party_label=r["party_label"], campaign_links=[], official_documents=[])
                    for r in dh[race]]
            src = dict(kind="secondary_source", url=DAILYHIVE_URL, publisher="Daily Hive",
                       published="2026-09-22",
                       note=("Official page vancouver.ca/your-government/2026-candidates-councillor.aspx "
                             "refused automated access (Cloudflare challenge) and has no archive capture; "
                             "list reproduced from Daily Hive's ballot-order article."))
            checks[race] = dict(official_count=None, dailyhive_count=len(rows),
                                note="no official capture available to cross-check")
            verification = "secondary_source"
        for r in rows:
            canon = PARTY_CANON.get(r["party_label"]) if r["party_label"] else None
            if r["party_label"] and not canon:
                canon = r["party_label"]
            cid = candidate_id(race, r["display_name"])
            candidates.append(dict(
                candidate_id=cid, race=race, office=info["label"], ballot_number=r["ballot_number"],
                display_name=r["display_name"], ballot_name=r["ballot_name"], name_cjk=r["name_cjk"],
                party_label_on_ballot=r["party_label"], party=canon or "No affiliation on ballot",
                party_id=PARTY_ID.get(canon, "none") if canon else "none",
                ballot_verification=verification, ballot_source=src,
                official_candidate_page=OFFICIAL_BASE + info["page"],
                campaign_links=r["campaign_links"], official_documents=r["official_documents"],
                affiliations=[], status="on_ballot_pre_election"))
    ebc = electionsbc_check(candidates)
    ids = [c["candidate_id"] for c in candidates]
    assert len(ids) == len(set(ids)), "duplicate candidate ids"
    meta = dict(
        retrieved_at=now_iso(), election_date="2026-10-17", phase="pre_election",
        phase_note=("All ballot data is pre-election (nominations closed; random ballot draw "
                    "2026-09-18). No results are included. Results published after polls close "
                    "must be added as a separate, dated record."),
        counts={r: sum(c["race"] == r for c in candidates) for r in RACES},
        total=len(candidates), seats={r: RACES[r]["seats"] for r in RACES},
        cross_checks=checks, elections_bc=ebc,
        expected_total_reported_by_dailyhive=130,
        inaccessible=[
            {"url": OFFICIAL_BASE + "2026-election.aspx", "reason": "Cloudflare bot challenge (HTTP 403)"},
            {"url": OFFICIAL_BASE + "2026-candidates-councillor.aspx",
             "reason": "Cloudflare bot challenge (HTTP 403); only archive capture is also a 403"},
            {"url": "https://vancouver.ca/files/cov/2026-vancouver-election-candidates-list.pdf",
             "reason": "Cloudflare bot challenge (HTTP 403); not archived"},
        ])
    (OUT / "candidates.json").write_text(json.dumps(candidates, indent=1, ensure_ascii=False))
    (OUT / "ballot_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, ensure_ascii=False))
