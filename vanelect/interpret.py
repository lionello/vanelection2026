"""Evidence-linked descriptions of principal components.

A principal component is a direction in the same embedding space as every
evidence chunk, so it can be described by the evidence itself rather than by
a label we invent:

  pole passages      verbatim chunks scoring furthest toward each end
                     (one per candidate, so no single campaign dominates)
  distinguishing     words over-represented in the top vs bottom quartile of
  terms              chunks (weighted log-odds with an informative Dirichlet
                     prior, Monroe, Colaresi & Quinn 2008)
  topic drivers      how much each topic's vectors vary along the axis, and,
                     per candidate, which of their topics push them which way
  stability          |cosine| between this axis and the same axis recomputed
                     with random source dropout; low means "don't read it"

Everything is reported as an AUTOMATED INTERPRETATION with citations. No
political label (left/right etc.) is ever assigned.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

STOP = set("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most my myself
no nor not now of off on once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too under until up very was
we were what when where which while who whom why will with would you your yours yourself yourselves us via per
city vancouver vancouver's council councillor candidate candidates party also new make ensure work including
within across every one two three year years s t ll re ve d m""".split())
TOKEN = re.compile(r"[a-z][a-z'\-]{2,}")


def tokens(text: str, banned: set[str]) -> list[str]:
    w = [t.strip("'-") for t in TOKEN.findall(text.lower())]
    w = [t for t in w if t and t not in STOP and t not in banned]
    return w + [f"{a} {b}" for a, b in zip(w, w[1:])]


def log_odds(top: list[str], bottom: list[str], prior: Counter, k: int = 10) -> tuple[list, list]:
    ca, cb = Counter(), Counter()
    for t in top:
        ca.update(t)
    for t in bottom:
        cb.update(t)
    a0 = sum(prior.values()) or 1
    scale = 500.0 / a0  # prior strength ~500 pseudo-counts
    na, nb = sum(ca.values()), sum(cb.values())
    res = []
    for w in set(ca) | set(cb):
        if ca[w] + cb[w] < 3:
            continue
        aw = prior[w] * scale + 0.01
        da = np.log((ca[w] + aw) / (na + 500 - ca[w] - aw))
        db = np.log((cb[w] + aw) / (nb + 500 - cb[w] - aw))
        var = 1 / (ca[w] + aw) + 1 / (cb[w] + aw)
        res.append((w, (da - db) / np.sqrt(var), ca[w], cb[w]))
    res.sort(key=lambda x: -x[1])
    pos = [dict(term=w, z=round(float(z), 2), n_pos=a, n_neg=b) for w, z, a, b in res[:k] if z > 1.96]
    neg = [dict(term=w, z=round(float(-z), 2), n_pos=a, n_neg=b) for w, z, a, b in res[::-1][:k] if z < -1.96]
    return pos, neg


def interpret_axes(st, race, view, mode, weights, cids, X, tvs, ids=None) -> list[dict]:
    from .analyze import pca_axes, candidate_vectors
    Y, evr, comps, mean = pca_axes(X)
    if not len(evr):
        return []
    topics = st.topics(view)
    # chunks that build these candidates' vectors (incl. party platform chunks in with_platform mode)
    subj = set(cids)
    if view == "platform" or (view == "policy" and mode == "with_platform"):
        subj |= {f"party:{st.cmap[c]['party_id']}" for c in cids}
    views = {view} | ({"platform"} if view == "policy" and mode == "with_platform" else set())
    idx = [i for v in views for i in st.by_view[v]
           if st.chunks[i]["subject"] in subj and not st.chunks[i].get("near_duplicate_of")
           and st.chunks[i]["topic"] in topics]
    banned = set()
    for c in st.cands:
        banned |= {w.lower() for w in re.findall(r"[A-Za-z]{3,}", c["display_name"])}
    toks = {i: tokens(st.chunks[i]["text"], banned) for i in idx}
    tmean = {t: st.vecs[[i for i in idx if st.chunks[i]["topic"] == t]].mean(0)
             for t in {st.chunks[i]["topic"] for i in idx}}
    prior = Counter()
    for t in toks.values():
        prior.update(t)
    # axis stability under source dropout
    cfg = st.cfg["sensitivity"]["source_dropout"]
    srcs = sorted({st.chunks[i]["source_id"] for i in idx})
    rng = np.random.default_rng(cfg["seed"] + 1)
    cos = [[], []]
    for _ in range(cfg["runs"]):
        drop = {s for s in srcs if rng.random() < cfg["drop_probability"]}
        ids2, X2, _, _ = candidate_vectors(st, race, view, mode, weights, exclude_sources=drop, ids=cids)
        if len(ids2) < 4:
            continue
        _, _, c2, _ = pca_axes(X2)
        for k in range(min(2, len(c2))):
            cos[k].append(abs(float(comps[k] @ c2[k])))
    axes = []
    for k in range(min(2, len(comps))):
        w = comps[k]
        if st.cfg.get("debias", {}).get("topic_centering"):
            # candidate vectors are built from topic-centred vectors, so score chunks the same way
            score = {i: float((st.vecs[i] - tmean[st.chunks[i]["topic"]]) @ w) for i in idx}
        else:
            score = {i: float((st.vecs[i] - mean) @ w) for i in idx}
        order = sorted(idx, key=lambda i: score[i])

        def poles(seq):
            out, seen = [], set()
            for i in seq:
                ch = st.chunks[i]
                if ch["subject"] in seen:
                    continue
                seen.add(ch["subject"])
                out.append(dict(subject=ch["subject"], topic=ch["topic"], text=ch["text"][:420],
                                source_id=ch["source_id"], passage_id=ch["passage_id"],
                                score=round(score[i], 4)))
                if len(out) >= 4:
                    break
            return out
        q = max(5, len(order) // 4)
        pos_terms, neg_terms = log_odds([toks[i] for i in order[-q:]], [toks[i] for i in order[:q]], prior)
        # topic drivers: spread of each topic's candidate vectors along the axis
        drivers = []
        for t, tname in enumerate(topics):
            vals = [float(tv[t] @ w) for tv, _ in (tvs[c] for c in cids) if not np.isnan(tv[t][0])]
            if len(vals) >= 3:
                drivers.append(dict(topic=tname, spread=round(float(np.std(vals)), 4), n=len(vals)))
        drivers.sort(key=lambda d: -d["spread"])
        # per-candidate pulls: this candidate's topic vector vs others' on the same topic
        topic_mean = {}
        for t in range(len(topics)):
            vals = [float(tvs[c][0][t] @ w) for c in cids if not np.isnan(tvs[c][0][t][0])]
            topic_mean[t] = (np.mean(vals), len(vals))
        pulls = {}
        for c in cids:
            tv, wt = tvs[c]
            m = ~np.isnan(tv[:, 0])
            share = (wt * m) / max((wt * m).sum(), 1e-9)
            items = []
            for t in np.where(m)[0]:
                mu, n = topic_mean[t]
                if n < 3:
                    continue
                items.append((topics[t], round(float(share[t] * (tv[t] @ w - mu)), 4)))
            items.sort(key=lambda x: -abs(x[1]))
            pulls[c] = items[:4]
        ranked = [cids[i] for i in np.argsort(Y[:, k])]
        stab = round(float(np.median(cos[k])), 3) if cos[k] else None
        gap = (evr[k] / evr[k + 1]) if k + 1 < len(evr) and evr[k + 1] else None
        axes.append(dict(
            component=k + 1, explained_variance=evr[k],
            stability_median_abs_cosine=stab,
            stable=bool(stab is not None and stab >= 0.8),
            eigen_ratio_to_next=round(float(gap), 3) if gap else None,
            positive_pole=dict(passages=poles(order[::-1]), terms=pos_terms, candidates=ranked[::-1][:3]),
            negative_pole=dict(passages=poles(order), terms=neg_terms, candidates=ranked[:3]),
            topic_drivers=drivers[:6], candidate_pulls=pulls,
            label="automated interpretation",
            note=("Described only by evidence: passages and words most aligned with each end of the axis. "
                  "No political label is implied. Read it only if 'stable' is true.")))
    return axes
