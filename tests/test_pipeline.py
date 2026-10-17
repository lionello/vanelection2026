"""Validation tests: identities, duplicates, citations, vote parsing, missing values.

Run after `make build`:  .venv/bin/pytest -q
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"


def jl(p):
    return [json.loads(l) for l in open(p)]


@pytest.fixture(scope="module")
def cands():
    return json.loads((P / "candidates.json").read_text())


@pytest.fixture(scope="module")
def claims():
    return jl(P / "claims.jsonl")


@pytest.fixture(scope="module")
def passages():
    return {p["passage_id"]: p for p in jl(P / "passages.jsonl")}


# ---------------------------------------------------------------- identities
def test_counts_and_ranges(cands):
    c = Counter(x["race"] for x in cands)
    assert c == {"mayor": 8, "council": 56, "park": 35, "school": 31}
    ranges = {"mayor": range(50, 58), "council": range(100, 156), "park": range(200, 235), "school": range(300, 331)}
    for race, rg in ranges.items():
        nums = sorted(x["ballot_number"] for x in cands if x["race"] == race)
        assert nums == list(rg), f"{race} ballot numbers not contiguous/unique"


def test_ids_unique_and_stable(cands):
    ids = [x["candidate_id"] for x in cands]
    assert len(ids) == len(set(ids))
    for x in cands:
        assert x["candidate_id"].startswith(x["race"] + "-")
        assert re.fullmatch(r"[a-z]+-[a-z0-9-]+", x["candidate_id"])


def test_every_candidate_officially_corroborated(cands):
    missing = [x["candidate_id"] for x in cands if not x.get("official_registration")]
    assert not missing, missing
    assert all(x["ballot_verification"] != "secondary_source" for x in cands)


def test_dual_candidacy_linked():
    site = json.loads((ROOT / "data" / "site" / "candidates.json").read_text())
    ch = [x for x in site if x["display_name"] == "Naomi Chocyk"]
    assert len(ch) == 2 and all(x["also_on_ballot_as"] for x in ch)


# ---------------------------------------------------------------- citations
def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def test_claims_are_verbatim_and_resolve(claims, passages):
    sources = {s["source_id"] for s in json.loads((P / "sources.json").read_text())}
    bad = []
    for c in claims:
        assert c["source_id"] in sources
        p = passages.get(c["passage_id"])
        assert p is not None, c["passage_id"]
        assert p["source_id"] == c["source_id"]
        if _norm(c["claim"]) not in _norm(p["text"]):
            bad.append(c["claim_id"])
    assert not bad, f"{len(bad)} claims not verbatim in their passage, e.g. {bad[:3]}"


def test_profile_citations_exist(claims):
    ids = {c["claim_id"] for c in claims}
    site = json.loads((ROOT / "data" / "site" / "candidates.json").read_text())
    for c in site:
        p = c["profile"]
        items = p["experience"] + p["actions"] + p["commentary"] + p["assessments"] + p["shared_language"] + [i for t in p["policy"].values() for i in t["items"]]
        for it in items:
            assert it["claim_id"] in ids
            assert it["url"], it


def test_party_commitments_never_personal(claims):
    for c in claims:
        if c["source_type"] == "party_platform":
            assert c["candidate_id"] is None and c["evidence_type"] == "party_commitment"
        if c["candidate_id"] is not None:
            assert c["evidence_type"] != "party_commitment"


def test_schema_fields(claims):
    need = {"candidate_id", "topic", "claim", "evidence_type", "source_id", "supporting_passage", "date",
            "verification_status", "uncertainty"}
    for c in claims[:500]:
        assert need <= set(c)


# ---------------------------------------------------------------- votes
def test_vote_values_preserved():
    d = pd.read_csv(P / "votes_candidates.csv")
    assert set(d.vote) <= {"In Favour", "In Opposition", "Absent", "Abstain", "No Vote", "Declared Conflict", "Ineligible"}
    vs = json.loads((P / "votes_summary.json").read_text())
    for cid, r in vs["summary"].items():
        assert sum(r["counts"].values()) == r["motions_recorded"]


def test_agreement_excludes_non_votes():
    """Overlap counted only on motions where BOTH cast For/Against; absence is never opposition."""
    vs = json.loads((P / "votes_summary.json").read_text())
    d = pd.read_csv(P / "votes_candidates.csv")
    d = d[d.contested]
    for pr in vs["pairs"][:10]:
        a = d[(d.candidate_id == pr["a"]) & d.vote.isin(["In Favour", "In Opposition"])].set_index("motion_id").vote
        b = d[(d.candidate_id == pr["b"]) & d.vote.isin(["In Favour", "In Opposition"])].set_index("motion_id").vote
        common = a.index.intersection(b.index)
        assert len(common) == pr["overlap_contested"]
        if len(common):
            assert abs((a[common] == b[common]).mean() - pr["agreement_contested"]) < 1e-3


def test_no_cross_term_comparison():
    vs = json.loads((P / "votes_summary.json").read_text())
    for pr in vs["pairs"]:
        if pr["overlap_contested"] > 0:
            assert pr["shared_terms"], pr


# ---------------------------------------------------------------- representations
def test_missing_topics_are_nan_not_zero():
    r = np.load(P / "repr_policy.npz")
    if len(r["subjects"]) == 0:
        pytest.skip("no policy evidence")
    v, m = r["vecs"], r["mask"]
    assert np.isnan(v[~m]).all()
    assert not np.isnan(v[m]).any()
    assert np.allclose(np.linalg.norm(v[m], axis=-1), 1, atol=1e-4)


def test_duplicate_document_does_not_change_representation():
    from vanelect import represent
    chunks = jl(P / "chunks.jsonl")
    vecs = np.load(P / "chunk_vectors.npy")
    subj = Counter(c["subject"] for c in chunks if c["view"] == "policy" and not c.get("near_duplicate_of"))
    if not subj:
        pytest.skip("no policy chunks")
    s = subj.most_common(1)[0][0]
    topics = json.loads(json.dumps(represent.config()["topics"]))
    base = represent.aggregate(chunks, vecs, "policy", topics)
    # simulate a syndicated copy: every chunk of s re-appears from a new source
    dup = [dict(c, source_id="SDUPLICATE", chunk_id=c["chunk_id"] + "d", near_duplicate_of=None)
           for c in chunks if c["subject"] == s and c["view"] == "policy"]
    all_c = [dict(c, near_duplicate_of=None) for c in chunks] + dup
    all_v = np.vstack([vecs, vecs[[i for i, c in enumerate(chunks) if c["subject"] == s and c["view"] == "policy"]]])
    represent.near_dedup(all_c, all_v, represent.config()["dedup"]["near_duplicate_cosine"])
    after = represent.aggregate(all_c, all_v, "policy", topics)
    i = base["subjects"].index(s)
    j = after["subjects"].index(s)
    m = base["mask"][i]
    assert (after["mask"][j] == m).all()
    cos = np.sum(base["vecs"][i][m] * after["vecs"][j][m], axis=-1)
    assert cos.min() > 0.999, cos
    assert all(c["near_duplicate_of"] for c in all_c[len(chunks):])


def test_features_missing_stays_missing():
    a = json.loads((ROOT / "data" / "site" / "analysis.json").read_text())
    for cid, f in a["features"].items():
        if f["years_elected_office"] is None:
            assert "no elected office documented" in f["years_basis"]
        assert f["management_responsibility"] in (None, "documented")


def test_board_votes_consistent():
    p = P / "board_votes.json"
    if not p.exists():
        pytest.skip("no board minutes parsed")
    b = json.loads(p.read_text())
    for m in b["motions"]:
        vals = set(m["members"].values())
        assert vals <= {"in_favour", "opposed", "absent", "abstained", "recused"}
        if m["result"].startswith("CARRIED UNANIMOUSLY"):
            assert "opposed" not in vals, m
        assert not m.get("parse_warning")
