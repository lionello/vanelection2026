"""Separate per-view, per-topic representations.

1. Chunk: consecutive canonical claims from the same passage, same subject,
   same view and same primary topic are joined (<= max_words_per_chunk).
2. Embed every chunk with the single configured model (L2-normalised).
3. Deduplicate: within a subject+view, chunks with cosine >= near_duplicate_cosine
   to an earlier chunk are dropped (repeated campaign language, syndication).
4. Aggregate: mean of a subject's chunks per topic, re-normalised -> topic vector.
   Topics without evidence stay MISSING (mask=False); they are never zero-filled.
5. Party platforms are embedded once per party (subject "party:<id>") and only
   linked to candidates at analysis time.

Output: data/processed/chunks.jsonl, data/processed/repr_<view>.npz
  npz: subjects[S], topics[T], vecs[S,T,D] (nan where missing), mask[S,T],
       counts[S,T] (chunks per topic)
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections import defaultdict

import numpy as np

from .embed import Embedder, config
from .fetch import ROOT

OUT = ROOT / "data" / "processed"


def view_of(evidence_type: str, cfg: dict) -> str | None:
    for v, spec in cfg["views"].items():
        if evidence_type in spec["evidence_types"]:
            return v
    return None


def make_chunks(claims: list[dict], cfg: dict) -> list[dict]:
    maxw = cfg["embedding"]["max_words_per_chunk"]
    topics = set(cfg["topics"])
    chunks, cur = [], None
    for c in claims:
        if not c["dedup_canonical"]:
            continue
        view = view_of(c["evidence_type"], cfg)
        if view is None:
            continue
        subj = c["candidate_id"] or (f"party:{c['party_id']}" if c["party_id"] else None)
        if subj is None:
            continue
        if view == "experience":
            topic = c["evidence_type"]
        else:
            topic = next((t for t in c["topics"] if t in topics), None)
            if topic is None:
                continue  # untopical sentences do not enter policy/platform/action vectors
        key = (subj, view, topic, c["passage_id"])
        words = len(c["claim"].split())
        if cur and cur["key"] == key and cur["words"] + words <= maxw:
            cur["text"] += " " + c["claim"]
            cur["claim_ids"].append(c["claim_id"])
            cur["words"] += words
        else:
            cur = dict(key=key, subject=subj, view=view, topic=topic, passage_id=c["passage_id"],
                       source_id=c["source_id"], text=c["claim"], claim_ids=[c["claim_id"]], words=words,
                       uncertainty=c["uncertainty"], evidence_type=c["evidence_type"])
            chunks.append(cur)
    for i, ch in enumerate(chunks):
        ch["chunk_id"] = f"K{i:06d}"
        del ch["key"]
    return chunks


def shared_across_party(chunks: list[dict], vecs: np.ndarray, thr: float, party_of: dict) -> None:
    """Policy chunks near-identical (cosine >= thr) to a chunk of ANOTHER candidate of the same party
    are shared campaign language: flagged and excluded from individual vectors."""
    idx = [i for i, c in enumerate(chunks) if c["view"] == "policy" and not c["subject"].startswith("party:")]
    by_party = defaultdict(list)
    for i in idx:
        by_party[party_of.get(chunks[i]["subject"], "none")].append(i)
    for pid, ii in by_party.items():
        if pid == "none" or len(ii) < 2:
            continue
        V = vecs[ii]
        S = V @ V.T
        for a, i in enumerate(ii):
            for b, j in enumerate(ii):
                if chunks[j]["subject"] != chunks[i]["subject"] and S[a, b] >= thr:
                    chunks[i]["shared_language_with"] = chunks[j]["subject"]
                    chunks[i]["near_duplicate_of"] = chunks[j]["chunk_id"]
                    chunks[i]["near_duplicate_cosine"] = round(float(S[a, b]), 4)
                    break


def near_dedup(chunks: list[dict], vecs: np.ndarray, thr: float) -> None:
    by = defaultdict(list)
    for i, ch in enumerate(chunks):
        by[(ch["subject"], ch["view"])].append(i)
    for idx in by.values():
        kept = []
        for i in idx:
            if kept:
                sims = vecs[kept] @ vecs[i]
                j = int(np.argmax(sims))
                if sims[j] >= thr:
                    chunks[i]["near_duplicate_of"] = chunks[kept[j]]["chunk_id"]
                    chunks[i]["near_duplicate_cosine"] = round(float(sims[j]), 4)
                    continue
            chunks[i]["near_duplicate_of"] = None
            kept.append(i)


def aggregate(chunks: list[dict], vecs: np.ndarray, view: str, topics: list[str],
              exclude_sources: set[str] = frozenset()) -> dict:
    subjects = sorted({c["subject"] for c in chunks if c["view"] == view})
    sidx = {s: i for i, s in enumerate(subjects)}
    tidx = {t: i for i, t in enumerate(topics)}
    D = vecs.shape[1]
    acc = np.zeros((len(subjects), len(topics), D), dtype=np.float64)
    cnt = np.zeros((len(subjects), len(topics)), dtype=int)
    for i, c in enumerate(chunks):
        if c["view"] != view or c.get("near_duplicate_of") or c["source_id"] in exclude_sources:
            continue
        if c["topic"] not in tidx:
            continue
        acc[sidx[c["subject"]], tidx[c["topic"]]] += vecs[i]
        cnt[sidx[c["subject"]], tidx[c["topic"]]] += 1
    mask = cnt > 0
    out = np.full_like(acc, np.nan)
    norms = np.linalg.norm(acc, axis=2, keepdims=True)
    out[mask] = (acc / np.where(norms == 0, 1, norms))[mask]
    return dict(subjects=subjects, topics=topics, vecs=out.astype(np.float32), mask=mask, counts=cnt)


def build() -> dict:
    cfg = config()
    claims = [json.loads(l) for l in (OUT / "claims.jsonl").open()]
    chunks = make_chunks(claims, cfg)
    emb = Embedder(cfg)
    vecs = emb.embed([c["text"] for c in chunks]) if chunks else np.zeros((0, 384), np.float32)
    near_dedup(chunks, vecs, cfg["dedup"]["near_duplicate_cosine"])
    party_of = {c["candidate_id"]: c["party_id"] for c in json.loads((OUT / "candidates.json").read_text())}
    shared_across_party(chunks, vecs, cfg["dedup"]["near_duplicate_cosine"], party_of)
    np.save(OUT / "chunk_vectors.npy", vecs)
    with (OUT / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    info = emb.model_info()
    stats = {}
    for view, spec in cfg["views"].items():
        topics = spec.get("topics", cfg["topics"])
        r = aggregate(chunks, vecs, view, topics)
        np.savez_compressed(OUT / f"repr_{view}.npz", subjects=np.array(r["subjects"]), topics=np.array(topics),
                            vecs=r["vecs"], mask=r["mask"], counts=r["counts"])
        stats[view] = dict(subjects=len(r["subjects"]), chunks=int(r["counts"].sum()))
    (OUT / "embedding_info.json").write_text(json.dumps(info, indent=1))
    return dict(chunks=len(chunks), near_duplicates=sum(1 for c in chunks if c.get("near_duplicate_of")),
                model=info, views=stats)


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
