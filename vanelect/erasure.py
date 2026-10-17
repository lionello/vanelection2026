"""Linear erasure of source/outlet style from chunk embeddings.

Nuisance directions are estimated from WITHIN-candidate contrasts: for every
candidate with evidence from >= 2 sources in a view, the mean of their chunks
from one source minus the mean of all their chunks in that view. Variation
along these directions is "the same person, written up by a different
source", which is source/style, not the candidate. The top principal
directions of these contrasts (enough to explain `variance_fraction`, capped
at `max_dims`) are projected out of every chunk in the view.

Effectiveness is measured, not assumed: leave-one-out nearest-centroid
accuracy at predicting a chunk's source type and publisher, before and after,
alongside the share of between-candidate variance that survives.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _loo_centroid_acc(X: np.ndarray, labels: list[str]) -> float | None:
    labs = np.array(labels)
    uniq = [u for u in set(labels) if (labs == u).sum() >= 5]
    keep = np.isin(labs, uniq)
    if len(uniq) < 2 or keep.sum() < 20:
        return None
    X, labs = X[keep], labs[keep]
    sums = {u: X[labs == u].sum(0) for u in uniq}
    cnts = {u: (labs == u).sum() for u in uniq}
    correct = 0
    for i in range(len(X)):
        best, bs = None, -9
        for u in uniq:
            s, n = sums[u], cnts[u]
            if u == labs[i]:
                s, n = s - X[i], n - 1
            c = s / n
            sc = float(X[i] @ c) / (np.linalg.norm(c) + 1e-12)
            if sc > bs:
                best, bs = u, sc
        correct += best == labs[i]
    return round(correct / len(X), 3)


def _between_candidate_share(X: np.ndarray, subj: list[str]) -> float:
    groups = defaultdict(list)
    for i, s in enumerate(subj):
        groups[s].append(i)
    mu = X.mean(0)
    between = sum(len(ii) * np.sum((X[ii].mean(0) - mu) ** 2) for ii in groups.values())
    total = np.sum((X - mu) ** 2)
    return float(between / total) if total else 0.0


def _within_candidate_source_acc(X: np.ndarray, subj: list[str], src: list[str]) -> float | None:
    """LOO nearest-centroid accuracy at telling a candidate's own sources apart (candidates with >= 2
    sources of >= 2 chunks), relative to chance; the confound erasure targets."""
    groups = defaultdict(list)
    for i, s_ in enumerate(subj):
        groups[s_].append(i)
    accs = []
    for ii in groups.values():
        labs = [src[i] for i in ii]
        cnt = defaultdict(int)
        for l in labs:
            cnt[l] += 1
        keep = [i for i in ii if cnt[src[i]] >= 2]
        if len({src[i] for i in keep}) < 2:
            continue
        a = _loo_centroid_acc_small(X[keep], [src[i] for i in keep])
        if a is not None:
            accs.append(a - 1 / len({src[i] for i in keep}))
    return round(float(np.mean(accs)), 3) if accs else None


def _loo_centroid_acc_small(X, labels):
    labs = np.array(labels)
    uniq = list(set(labels))
    correct = 0
    for i in range(len(X)):
        best, bs = None, -9
        for u in uniq:
            m = labs == u
            m[i] = False
            if not m.any():
                continue
            c = X[m].mean(0)
            sc = float(X[i] @ c) / (np.linalg.norm(c) + 1e-12)
            if sc > bs:
                best, bs = u, sc
        correct += best == labs[i]
    return correct / len(X)


def erase_sources(chunks: list[dict], vecs: np.ndarray, cfg: dict, raw: np.ndarray | None = None,
                  publisher: dict | None = None) -> tuple[np.ndarray, dict]:
    out = vecs.astype(np.float64).copy()
    report = {}
    for view in sorted({c["view"] for c in chunks}):
        idx = [i for i, c in enumerate(chunks) if c["view"] == view and not c.get("near_duplicate_of")]
        cand_idx = [i for i in idx if not chunks[i]["subject"].startswith("party:")]
        by_subj = defaultdict(lambda: defaultdict(list))
        for i in cand_idx:
            by_subj[chunks[i]["subject"]][chunks[i]["source_id"]].append(i)
        R = []
        for subj, srcs in by_subj.items():
            if len(srcs) < 2:
                continue
            m_all = vecs[[i for ii in srcs.values() for i in ii]].mean(0)
            for ii in srcs.values():
                R.append(vecs[ii].mean(0) - m_all)
        if len(R) < 10:
            report[view] = dict(erased_dims=0, note="too few within-candidate source contrasts")
            continue
        R = np.array(R)
        U, S, Vt = np.linalg.svd(R - R.mean(0) * 0, full_matrices=False)
        var = S ** 2 / (S ** 2).sum()
        k = int(np.searchsorted(np.cumsum(var), cfg.get("variance_fraction", 0.5)) + 1)
        k = min(k, cfg.get("max_dims", 24), len(S))
        B = Vt[:k]  # (k, D) orthonormal
        all_idx = [i for i, c in enumerate(chunks) if c["view"] == view]
        before = vecs[idx]
        out[all_idx] = out[all_idx] - (out[all_idx] @ B.T) @ B
        out[all_idx] /= np.linalg.norm(out[all_idx], axis=1, keepdims=True).clip(1e-12)
        after = out[idx].astype(np.float32)
        st = [chunks[i].get("genre", "?") for i in idx]
        pub = [(publisher or {}).get(chunks[i]["source_id"], chunks[i]["source_id"]) for i in idx]
        src = [chunks[i]["source_id"] for i in idx]
        subj = [chunks[i]["subject"] for i in idx]
        stages = {"raw": raw[idx] if raw is not None else before, "genre_centred": before, "erased": after}
        rep = dict(erased_dims=k, contrasts=len(R), contrast_variance_erased=round(float(var[:k].sum()), 3),
                   chance_source_type=round(1 / max(1, len(set(st))), 3), stages={})
        for name, X in stages.items():
            rep["stages"][name] = dict(
                source_type_acc=_loo_centroid_acc(X, st), publisher_acc=_loo_centroid_acc(X, pub),
                within_candidate_source_acc_over_chance=_within_candidate_source_acc(X, subj, src),
                between_candidate_variance_share=round(_between_candidate_share(X, subj), 3))
        report[view] = rep
    return out.astype(np.float32), report
