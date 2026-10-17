"""Council recorded votes from the City of Vancouver open-data portal.

Dataset: council-voting-records (Open Government Licence - Vancouver).
The City's note: "Please consult corresponding meeting minutes for official
vote records" - so every vote here is labelled `official_open_data` and links
to the dataset; minutes remain the authoritative record.

Vote values are kept exactly as published and are NOT collapsed:
  In Favour | In Opposition | Absent | Abstain | No Vote | Declared Conflict | Ineligible
Only In Favour / In Opposition count as votes for agreement statistics.
Absent, Abstain, No Vote, Declared Conflict (recusal) and Ineligible are
non-votes and are never treated as opposition.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import ROOT, now_iso

CSV = ROOT / "data" / "official" / "council-voting-records.csv"
DATASET_URL = "https://opendata.vancouver.ca/explore/dataset/council-voting-records/"
EXPORT_URL = ("https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/"
              "council-voting-records/exports/csv?delimiter=%2C")
OUT = ROOT / "data" / "processed"

# council_member label in dataset -> candidate ids (one person may hold several
# ballot entries, but only council/mayor candidates sat on Council).
MEMBER_MAP = {
    "Mayor K Sim": ["mayor-ken-sim"],
    "Councillor P Fry": ["mayor-pete-fry"],
    "Councillor R Bligh": ["mayor-rebecca-bligh"],
    "Councillor C Hardwick": ["mayor-colleen-hardwick"],
    "Councillor S Kirby-Yung": ["council-sarah-kirby-yung"],
    "Councillor L Dominato": ["council-lisa-dominato"],
    "Councillor M Klassen": ["council-mike-klassen"],
    "Councillor P Meiszner": ["council-peter-meiszner"],
    "Councillor L Zhou": ["council-lenny-zhou"],
    "Councillor S Orr": ["council-sean-orr"],
    "Councillor L Maloney": ["council-lucy-maloney"],
}
CAST = {"In Favour": 1, "In Opposition": -1}
NON_VOTES = ["Absent", "Abstain", "No Vote", "Declared Conflict", "Ineligible"]
TERMS = [("2016-2018 (part)", "2016-01-01", "2018-11-04"), ("2018-2022", "2018-11-05", "2022-11-06"),
         ("2022-2026", "2022-11-07", "2026-12-31")]
MIN_OVERLAP = 100  # matching contested motions required for a pairwise comparison


def fix_mojibake(s: str) -> str:
    if not isinstance(s, str):
        return s
    try:
        if "â" in s or "Ã" in s:
            return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return s


def load() -> pd.DataFrame:
    d = pd.read_csv(CSV)
    d["agenda_description"] = d["agenda_description"].map(fix_mojibake)
    d["vote_date"] = pd.to_datetime(d["vote_date"])
    # A motion is identified by (meeting_id, vote_number); both members must have
    # voted on that exact motion to be compared.
    d["motion_id"] = d["meeting_id"].astype(str) + ":" + d["vote_number"].astype(str)
    d["contested"] = d["decision"] != "Carried Unanimously"
    return d


def term_of(date) -> str:
    for name, a, b in TERMS:
        if pd.Timestamp(a) <= date <= pd.Timestamp(b):
            return name
    return "unknown"


def ensure_inputs() -> None:
    import requests
    if not CSV.exists():
        CSV.parent.mkdir(parents=True, exist_ok=True)
        CSV.write_bytes(requests.get(EXPORT_URL, timeout=300).content)
        CSV.with_suffix(".meta.json").write_bytes(requests.get(
            "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/council-voting-records", timeout=60).content)


def build() -> dict:
    ensure_inputs()
    d = load()
    sha = hashlib.sha256(CSV.read_bytes()).hexdigest()
    meta = json.loads((CSV.with_suffix(".meta.json")).read_text()) if CSV.with_suffix(".meta.json").exists() else {}
    modified = meta.get("metas", {}).get("default", {}).get("modified")
    ours = d[d.council_member.isin(MEMBER_MAP)].copy()
    # --- per-candidate record summary -------------------------------------------------
    summary = {}
    for member, cids in MEMBER_MAP.items():
        m = ours[ours.council_member == member]
        counts = m.vote.value_counts().to_dict()
        contested = m[m.contested]
        terms = sorted({term_of(x) for x in m.vote_date})
        rec = dict(
            council_member_label=member, first_vote=str(m.vote_date.min().date()),
            last_vote=str(m.vote_date.max().date()), motions_recorded=int(len(m)),
            counts={k: int(counts.get(k, 0)) for k in list(CAST) + NON_VOTES},
            contested_motions=int(len(contested)),
            contested_counts={k: int(v) for k, v in contested.vote.value_counts().items()},
            absence_rate=round(counts.get("Absent", 0) / len(m), 4),
            declared_conflicts=int(counts.get("Declared Conflict", 0)),
            terms=terms, source_url=DATASET_URL)
        for cid in cids:
            summary[cid] = rec
    # --- pairwise agreement on matching contested motions ----------------------------
    cast = ours[ours.vote.isin(CAST)].copy()
    cast["v"] = cast.vote.map(CAST)
    pivot_all = cast.pivot_table(index="motion_id", columns="council_member", values="v", aggfunc="first")
    contested_ids = set(ours.loc[ours.contested, "motion_id"])
    pivot_c = pivot_all.loc[pivot_all.index.isin(contested_ids)]
    pairs = []
    for a, b in combinations(MEMBER_MAP, 2):
        if a not in pivot_c or b not in pivot_c:
            continue
        both = pivot_c[[a, b]].dropna()
        n = len(both)
        rec = dict(a=MEMBER_MAP[a][0], b=MEMBER_MAP[b][0], overlap_contested=int(n),
                   sufficient=n >= MIN_OVERLAP)
        if n:
            rec["agreement_contested"] = round(float((both[a] == both[b]).mean()), 4)
        bothall = pivot_all[[a, b]].dropna()
        rec["overlap_all"] = int(len(bothall))
        if len(bothall):
            rec["agreement_all"] = round(float((bothall[a] == bothall[b]).mean()), 4)
        # shared terms only: comparison never spans different councils
        rec["shared_terms"] = sorted(set(summary[MEMBER_MAP[a][0]]["terms"]) & set(summary[MEMBER_MAP[b][0]]["terms"]))
        pairs.append(rec)
    # --- per-candidate vote rows (contested only, for evidence display) ---------------
    rows = []
    for _, r in ours.iterrows():
        rows.append(dict(candidate_id=MEMBER_MAP[r.council_member][0], motion_id=r.motion_id,
                         meeting_id=int(r.meeting_id), meeting_type=r.meeting_type,
                         date=str(r.vote_date.date()), motion=r.agenda_description, vote=r.vote,
                         decision=r.decision, contested=bool(r.contested),
                         vote_detail_id=int(r.vote_detail_id)))
    out = dict(dataset=dict(url=DATASET_URL, export_url=EXPORT_URL, sha256=sha, modified=modified,
                            retrieved_at=now_iso(), publisher="City of Vancouver",
                            licence="Open Government Licence - Vancouver",
                            caveat="City note: consult meeting minutes for official vote records."),
               member_map=MEMBER_MAP, min_overlap=MIN_OVERLAP, summary=summary, pairs=pairs,
               value_semantics={"In Favour": "vote for", "In Opposition": "vote against",
                                "Absent": "not present - NOT a vote against",
                                "Abstain": "present, did not vote - NOT a vote against",
                                "No Vote": "no vote recorded - NOT a vote against",
                                "Declared Conflict": "recused (conflict of interest) - NOT a vote against",
                                "Ineligible": "not eligible (e.g. missed public hearing) - NOT a vote against"})
    out["abc_caucus_departures"] = caucus_departures()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "votes_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(OUT / "votes_candidates.csv", index=False)
    return out


# ABC council caucus membership 2022-2026 (the only caucus with >= 3 members). Dates are documented:
# elected 2022-11-07 (City inaugural meeting); Bligh expelled 2025-02-14 (Global News, see URL).
ABC_CAUCUS = {
    "Mayor K Sim": ("2022-11-07", None), "Councillor S Kirby-Yung": ("2022-11-07", None),
    "Councillor L Dominato": ("2022-11-07", None), "Councillor M Klassen": ("2022-11-07", None),
    "Councillor P Meiszner": ("2022-11-07", None), "Councillor L Zhou": ("2022-11-07", None),
    "Councillor B Montague": ("2022-11-07", None),
    "Councillor R Bligh": ("2022-11-07", "2025-02-13"),
}
CAUCUS_SOURCES = ["https://globalnews.ca/news/11020156/abc-expels-rebecca-bligh",
                  "https://en.wikipedia.org/wiki/2022_Vancouver_municipal_election"]


def caucus_departures() -> dict:
    """Contested motions where an ABC caucus member cast the opposite vote to the majority of the
    other ABC members who cast a vote. Non-votes are excluded on both sides. Evidence only: a
    departure is not an endorsement of anyone else's position."""
    d = load()
    d = d[d.contested & d.vote.isin(CAST) & d.council_member.isin(ABC_CAUCUS)]
    def member_on(row):
        a, b = ABC_CAUCUS[row.council_member]
        return pd.Timestamp(a) <= row.vote_date and (b is None or row.vote_date <= pd.Timestamp(b))
    d = d[d.apply(member_on, axis=1)]
    out = {}
    for mid, g in d.groupby("motion_id"):
        for _, r in g.iterrows():
            others = g[g.council_member != r.council_member]
            if len(others) < 3:
                continue
            fav = (others.vote == "In Favour").sum()
            opp = (others.vote == "In Opposition").sum()
            if fav == opp:
                continue
            maj = "In Favour" if fav > opp else "In Opposition"
            if r.vote != maj and r.council_member in MEMBER_MAP:
                cid = MEMBER_MAP[r.council_member][0]
                out.setdefault(cid, []).append(dict(motion_id=mid, date=str(r.vote_date.date()), motion=r.agenda_description,
                                                    vote=r.vote, caucus_majority=maj, caucus_split=f"{fav}-{opp}",
                                                    decision=r.decision))
    tot = d.groupby("council_member").motion_id.nunique()
    return dict(departures=out,
                contested_votes_as_caucus_member={MEMBER_MAP[m][0]: int(n) for m, n in tot.items() if m in MEMBER_MAP},
                caucus=ABC_CAUCUS, sources=CAUCUS_SOURCES,
                method="Contested motions only; majority of the other ABC members casting For/Against; ties skipped; "
                       "membership dates applied per vote.")


def vote_matrix(term: str = "2022-2026", contested_only: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """motion x candidate matrix of +1/-1/NaN for a single council term."""
    d = load()
    ours = d[d.council_member.isin(MEMBER_MAP)].copy()
    ours = ours[ours.vote_date.map(term_of) == term]
    if contested_only:
        ours = ours[ours.contested]
    ours["v"] = ours.vote.map(CAST)
    p = ours.pivot_table(index="motion_id", columns="council_member", values="v", aggfunc="first")
    p.columns = [MEMBER_MAP[c][0] for c in p.columns]
    return p, list(p.columns)


def agreement_embedding(term: str = "2022-2026") -> dict:
    """Classical MDS of pairwise disagreement on matching contested motions in one term."""
    p, cids = vote_matrix(term)
    k = len(cids)
    A = np.full((k, k), np.nan)
    N = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            both = p[[cids[i], cids[j]]].dropna()
            N[i, j] = len(both)
            if len(both) >= MIN_OVERLAP:
                A[i, j] = (both.iloc[:, 0] == both.iloc[:, 1]).mean()
    keep = [i for i in range(k) if np.all(~np.isnan(A[i, [j for j in range(k)]]))]
    D = 1 - A[np.ix_(keep, keep)]
    n = len(keep)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    pos = np.clip(w, 0, None)
    coords = V[:, :2] * np.sqrt(pos[:2])
    return dict(term=term, candidates=[cids[i] for i in keep], coords=coords.tolist(),
                explained=(pos[:2] / pos.sum()).round(4).tolist() if pos.sum() else [0, 0],
                agreement=A[np.ix_(keep, keep)].round(4).tolist(), overlap=N[np.ix_(keep, keep)].tolist(),
                excluded=[cids[i] for i in range(k) if i not in keep],
                method="classical MDS on (1 - agreement) over matching contested motions; "
                       f"pairs need >= {MIN_OVERLAP} shared cast votes")


if __name__ == "__main__":
    o = build()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('first_vote', 'last_vote', 'motions_recorded', 'absence_rate', 'counts')} for k, v in o["summary"].items()}, indent=1))
    for t in ["2018-2022", "2022-2026"]:
        e = agreement_embedding(t)
        print(t, e["candidates"], e["explained"], e["excluded"])
