"""Unsupervised analysis per race and per view.

* PCA (primary, reproducible, linear): 2-D coordinates + explained variance.
* UMAP (optional, nonlinear): disclosed settings and seed; axes carry no meaning.
* Cosine nearest neighbours in the ORIGINAL embedding space, with the most
  similar supporting chunk pairs as evidence.
* Sensitivity: source dropout, topic re-weighting, party-platform toggle,
  UMAP seeds, and projected-vs-original neighbour agreement.

No labels, no supervised models, no axis interpretation is produced here.
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA

from .embed import config
from .fetch import ROOT, now_iso
from . import votes as votes_mod

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUT = ROOT / "data" / "processed"
SITE = ROOT / "data" / "site"
RACES = ["mayor", "council", "park", "school"]


# ------------------------------------------------------------------ state
class State:
    def __init__(self):
        self.cfg = config()
        self.cands = json.loads((OUT / "candidates.json").read_text())
        self.cmap = {c["candidate_id"]: c for c in self.cands}
        self.chunks = [json.loads(l) for l in (OUT / "chunks.jsonl").open()]
        self.vecs = np.load(OUT / "chunk_vectors.npy")
        self.by_view = defaultdict(list)
        for i, c in enumerate(self.chunks):
            self.by_view[c["view"]].append(i)

    def topics(self, view):
        return self.cfg["views"][view].get("topics", self.cfg["topics"])

    def topic_matrix(self, view, exclude_sources=frozenset()):
        """subject -> (T,D) topic vectors (nan = missing), counts (T,)"""
        topics = self.topics(view)
        tidx = {t: i for i, t in enumerate(topics)}
        D = self.vecs.shape[1]
        acc, cnt = {}, {}
        for i in self.by_view[view]:
            c = self.chunks[i]
            if c.get("near_duplicate_of") or c["source_id"] in exclude_sources or c["topic"] not in tidx:
                continue
            s = c["subject"]
            if s not in acc:
                acc[s] = np.zeros((len(topics), D))
                cnt[s] = np.zeros(len(topics), dtype=int)
            acc[s][tidx[c["topic"]]] += self.vecs[i]
            cnt[s][tidx[c["topic"]]] += 1
        out = {}
        for s in acc:
            m = cnt[s] > 0
            v = np.full_like(acc[s], np.nan)
            v[m] = acc[s][m] / np.linalg.norm(acc[s][m], axis=1, keepdims=True)
            out[s] = (v, cnt[s])
        return out


def combine_topics(tv: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    m = ~np.isnan(tv[:, 0])
    w = weights * m
    if w.sum() <= 0:
        return None
    v = np.nansum(tv * w[:, None], axis=0) / w.sum()
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def candidate_vectors(st: State, race: str | None, view: str, mode: str = "individual",
                      weights: dict | None = None, exclude_sources=frozenset(), ids: list[str] | None = None):
    """Returns ids, X, info{id: {...coverage...}}, insufficient{id: reason}."""
    cfg = st.cfg
    thr = cfg["thresholds"]
    topics = st.topics(view if view != "policy" or True else view)
    w = np.array([float((weights or {}).get(t, 1.0)) for t in st.topics(view)])
    tm = st.topic_matrix(view, exclude_sources)
    pm = st.topic_matrix("platform", exclude_sources) if (view == "policy" and mode == "with_platform") else {}
    pool = ids or [c["candidate_id"] for c in st.cands if race is None or c["race"] == race]
    out_ids, X, info, insufficient = [], [], {}, {}
    for cid in pool:
        c = st.cmap[cid]
        tv, cnt = tm.get(cid, (np.full((len(topics), st.vecs.shape[1]), np.nan), np.zeros(len(topics), int)))
        tv = tv.copy()
        platform_topics = []
        if view == "platform":
            ptv = tm.get(f"party:{c['party_id']}")
            if ptv is None:
                insufficient[cid] = "no party platform collected" if c["party_id"] != "none" else "no party (independent)"
                continue
            tv, cnt = ptv
        elif pm:
            ptv = pm.get(f"party:{c['party_id']}")
            if ptv is not None:
                for t in range(len(topics)):
                    a, b = tv[t], ptv[0][t]
                    if np.isnan(b[0]):
                        continue
                    platform_topics.append(st.topics(view)[t])
                    if np.isnan(a[0]):
                        tv[t] = b
                    else:
                        s = a + b
                        tv[t] = s / np.linalg.norm(s)
        n_chunks = int(cnt.sum())
        covered = [st.topics(view)[t] for t in range(len(topics)) if not np.isnan(tv[t][0])]
        info[cid] = dict(chunks=n_chunks, topics_covered=covered, platform_topics=platform_topics)
        if view != "platform" and n_chunks < thr["min_chunks"] and not platform_topics:
            insufficient[cid] = f"only {n_chunks} evidence chunk(s) (< {thr['min_chunks']})"
            continue
        if view in ("policy", "actions") and len(covered) < thr["min_topics"]:
            insufficient[cid] = f"only {len(covered)} topic(s) covered (< {thr['min_topics']})"
            continue
        v = combine_topics(tv, w)
        if v is None:
            insufficient[cid] = "all evidence in zero-weighted topics"
            continue
        out_ids.append(cid)
        X.append(v)
    return out_ids, (np.array(X) if X else np.zeros((0, st.vecs.shape[1]))), info, insufficient


# ------------------------------------------------------------------ projections
def pca2(X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    n = len(X)
    if n < 3:
        return np.zeros((n, 2)), []
    p = PCA(n_components=min(n - 1, X.shape[1], 10), random_state=0).fit(X)
    Y = p.transform(X)[:, :2]
    # deterministic sign: largest-|loading| coordinate positive
    for k in range(Y.shape[1]):
        if Y[np.argmax(np.abs(Y[:, k])), k] < 0:
            Y[:, k] *= -1
    return Y, [round(float(x), 4) for x in p.explained_variance_ratio_]


def umap2(X: np.ndarray, seed: int, cfg: dict) -> np.ndarray | None:
    n = len(X)
    if n < 6:
        return None
    import umap
    nn = max(2, min(cfg["n_neighbors"], n - 1))
    r = umap.UMAP(n_neighbors=nn, min_dist=cfg["min_dist"], metric=cfg["metric"], random_state=seed,
                  init="spectral" if n > nn + 1 else "random", n_jobs=1)
    return r.fit_transform(X)


def procrustes_disp(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Align B to A (rotation/reflection + scale); per-point displacement / RMS spread of A."""
    A0, B0 = A - A.mean(0), B - B.mean(0)
    sa = np.sqrt((A0 ** 2).sum(1).mean()) or 1.0
    sb = np.sqrt((B0 ** 2).sum(1).mean()) or 1.0
    A1, B1 = A0 / sa, B0 / sb
    U, _, Vt = np.linalg.svd(B1.T @ A1)
    R = U @ Vt
    return np.linalg.norm(A1 - B1 @ R, axis=1)


def knn(X: np.ndarray, k: int) -> list[list[int]]:
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    return [list(np.argsort(-S[i])[:k]) for i in range(len(X))]


def knn2d(Y: np.ndarray, k: int) -> list[list[int]]:
    D = ((Y[:, None] - Y[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return [list(np.argsort(D[i])[:k]) for i in range(len(Y))]


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


# ------------------------------------------------------------------ evidence for neighbours
def support_pairs(st: State, view: str, a: str, b: str, mode: str, top: int = 2) -> list[dict]:
    def chunks_of(s):
        subj = [s]
        if view == "policy" and mode == "with_platform":
            subj.append(f"party:{st.cmap[s]['party_id']}")
        v = "platform" if view == "platform" else view
        if view == "platform":
            subj = [f"party:{st.cmap[s]['party_id']}"]
        return [i for i in st.by_view[v] if st.chunks[i]["subject"] in subj and not st.chunks[i].get("near_duplicate_of")] + \
               ([i for i in st.by_view["platform"] if st.chunks[i]["subject"] in subj[1:] and not st.chunks[i].get("near_duplicate_of")]
                if view == "policy" and mode == "with_platform" else [])
    ia, ib = chunks_of(a), chunks_of(b)
    if not ia or not ib:
        return []
    S = st.vecs[ia] @ st.vecs[ib].T
    out, used = [], set()
    for flat in np.argsort(-S, axis=None):
        i, j = np.unravel_index(flat, S.shape)
        ca, cb = st.chunks[ia[i]], st.chunks[ib[j]]
        if ca["topic"] != cb["topic"] or ca["topic"] in used:
            continue
        used.add(ca["topic"])
        out.append(dict(topic=ca["topic"], cosine=round(float(S[i, j]), 3),
                        a=dict(chunk_id=ca["chunk_id"], text=ca["text"][:400], passage_id=ca["passage_id"],
                               source_id=ca["source_id"], subject=ca["subject"]),
                        b=dict(chunk_id=cb["chunk_id"], text=cb["text"][:400], passage_id=cb["passage_id"],
                               source_id=cb["source_id"], subject=cb["subject"])))
        if len(out) >= top:
            break
    return out


# ------------------------------------------------------------------ one map
def analyse_map(st: State, race: str | None, view: str, mode: str = "individual",
                weights: dict | None = None, sensitivity: bool = True, ids=None) -> dict:
    cfg = st.cfg
    k = cfg["thresholds"]["neighbours_k"]
    cids, X, info, insufficient = candidate_vectors(st, race, view, mode, weights, ids=ids)
    res = dict(race=race, view=view, mode=mode, n=len(cids), insufficient=insufficient, coverage=info,
               generated_at=now_iso())
    if len(cids) < 3:
        res["note"] = "fewer than 3 candidates with sufficient evidence - no projection"
        res["points"] = [dict(id=c) for c in cids]
        return res
    Y, evr = pca2(X)
    res["pca"] = dict(explained_variance_ratio=evr,
                      note="PCA on L2-normalised candidate vectors (topic-equalised). Axes are directions "
                           "of greatest variance in the embedding space; they carry no political meaning.")
    U = umap2(X, cfg["umap"]["seeds"][0], cfg["umap"])
    res["umap"] = dict(available=U is not None, seed=cfg["umap"]["seeds"][0],
                       n_neighbors=min(cfg["umap"]["n_neighbors"], len(cids) - 1),
                       min_dist=cfg["umap"]["min_dist"], metric=cfg["umap"]["metric"],
                       note="UMAP is nonlinear and stochastic; distances between clusters and the axes "
                            "have no inherent meaning." if U is not None else "UMAP needs at least 6 candidates")
    kk = min(k, len(cids) - 1)
    nb = knn(X, kk)
    S = X @ X.T
    pts = []
    for i, cid in enumerate(cids):
        pts.append(dict(id=cid, pca=[round(float(x), 5) for x in Y[i]],
                        umap=[round(float(x), 5) for x in U[i]] if U is not None else None,
                        neighbours=[dict(id=cids[j], cosine=round(float(S[i, j]), 4)) for j in nb[i]]))
    res["points"] = pts
    same = [np.mean([st.cmap[n["id"]]["party_id"] == st.cmap[p["id"]]["party_id"] for n in p["neighbours"][:3]])
            for p in pts if st.cmap[p["id"]]["party_id"] != "none"
            and sum(st.cmap[q["id"]]["party_id"] == st.cmap[p["id"]]["party_id"] for q in pts) > 1]
    res["same_party_neighbour_rate"] = round(float(np.mean(same)), 3) if same else None
    if sensitivity:
        res["stability"] = stability(st, race, view, mode, weights, cids, X, Y, U, ids=ids)
    return res


def stability(st, race, view, mode, weights, cids, X, Y, U, ids=None) -> dict:
    cfg = st.cfg["sensitivity"]
    n = len(cids)
    k3 = max(1, min(3, n - 2))
    base_nb = knn(X, k3)
    per = {c: dict(dropout_disp=[], dropout_jacc=[], weight_disp=[], weight_jacc=[], vanished=0) for c in cids}
    # --- source dropout
    srcs = sorted({st.chunks[i]["source_id"] for i in range(len(st.chunks))
                   if st.chunks[i]["subject"] in set(cids) | {f"party:{st.cmap[c]['party_id']}" for c in cids}})
    rng = np.random.default_rng(cfg["source_dropout"]["seed"])
    for _ in range(cfg["source_dropout"]["runs"]):
        drop = {s for s in srcs if rng.random() < cfg["source_dropout"]["drop_probability"]}
        ids2, X2, _, _ = candidate_vectors(st, race, view, mode, weights, exclude_sources=drop, ids=cids)
        keep = [i for i, c in enumerate(cids) if c in ids2]
        for i, c in enumerate(cids):
            if c not in ids2:
                per[c]["vanished"] += 1
        if len(keep) < 4:
            continue
        pos = {c: j for j, c in enumerate(ids2)}
        Xs = np.array([X2[pos[cids[i]]] for i in keep])
        Y2, _ = pca2(Xs)
        d = procrustes_disp(Y[keep], Y2)
        nb2 = knn(Xs, min(k3, len(keep) - 2) or 1)
        for a, i in enumerate(keep):
            per[cids[i]]["dropout_disp"].append(float(d[a]))
            per[cids[i]]["dropout_jacc"].append(jacc({keep[x] for x in nb2[a]}, base_nb[i]))
    # --- topic weights (views with policy topics)
    topics = st.topics(view)
    rng = np.random.default_rng(cfg["topic_weights"]["seed"])
    for _ in range(cfg["topic_weights"]["runs"]):
        w = rng.dirichlet([cfg["topic_weights"]["dirichlet_alpha"]] * len(topics)) * len(topics)
        ids2, X2, _, _ = candidate_vectors(st, race, view, mode, dict(zip(topics, w)), ids=cids)
        if ids2 != cids:
            keep = [i for i, c in enumerate(cids) if c in ids2]
            pos = {c: j for j, c in enumerate(ids2)}
            X2 = np.array([X2[pos[cids[i]]] for i in keep])
        else:
            keep = list(range(n))
        if len(keep) < 4:
            continue
        Y2, _ = pca2(X2)
        d = procrustes_disp(Y[keep], Y2)
        nb2 = knn(X2, min(k3, len(keep) - 2) or 1)
        for a, i in enumerate(keep):
            per[cids[i]]["weight_disp"].append(float(d[a]))
            per[cids[i]]["weight_jacc"].append(jacc({keep[x] for x in nb2[a]}, base_nb[i]))
    # --- UMAP seeds
    umap_disp = {c: [] for c in cids}
    if U is not None:
        for seed in st.cfg["umap"]["seeds"][1:]:
            U2 = umap2(X, seed, st.cfg["umap"])
            d = procrustes_disp(U, U2)
            for i, c in enumerate(cids):
                umap_disp[c].append(float(d[i]))
    # --- projection fidelity: 2-D neighbours vs original-space neighbours
    nb_pca = knn2d(Y, k3)
    nb_umap = knn2d(U, k3) if U is not None else None
    # --- platform toggle (policy view only)
    toggle = {}
    if view == "policy":
        other = "with_platform" if mode == "individual" else "individual"
        ids2, X2, _, _ = candidate_vectors(st, race, view, other, weights, ids=cids)
        common = [c for c in cids if c in ids2]
        if len(common) >= 4:
            ia = [cids.index(c) for c in common]
            ib = [ids2.index(c) for c in common]
            nba, nbb = knn(X[ia], min(k3, len(common) - 2) or 1), knn(X2[ib], min(k3, len(common) - 2) or 1)
            for a, c in enumerate(common):
                toggle[c] = round(jacc({common[x] for x in nba[a]}, {common[x] for x in nbb[a]}), 3)
    thr_d, thr_j = cfg["unstable_displacement"], cfg["unstable_neighbour_jaccard"]
    out = {}
    for i, c in enumerate(cids):
        p = per[c]
        med = lambda xs: round(float(np.median(xs)), 3) if xs else None
        rec = dict(dropout_displacement=med(p["dropout_disp"]), dropout_neighbour_jaccard=med(p["dropout_jacc"]),
                   dropout_runs_without_evidence=p["vanished"],
                   weight_displacement=med(p["weight_disp"]), weight_neighbour_jaccard=med(p["weight_jacc"]),
                   umap_seed_displacement=med(umap_disp[c]),
                   pca_neighbour_agreement=round(jacc(nb_pca[i], base_nb[i]), 3),
                   umap_neighbour_agreement=round(jacc(nb_umap[i], base_nb[i]), 3) if nb_umap else None,
                   platform_toggle_neighbour_jaccard=toggle.get(c))
        reasons = []
        if rec["dropout_displacement"] is not None and rec["dropout_displacement"] > thr_d:
            reasons.append("moves under source removal")
        if rec["dropout_neighbour_jaccard"] is not None and rec["dropout_neighbour_jaccard"] < thr_j:
            reasons.append("neighbours change under source removal")
        if rec["weight_displacement"] is not None and rec["weight_displacement"] > thr_d:
            reasons.append("moves under topic re-weighting")
        if p["vanished"] > cfg["source_dropout"]["runs"] * 0.2:
            reasons.append("evidence depends on few sources")
        rec["unstable"] = bool(reasons)
        # projection fidelity is reported separately: it is a property of the 2-D view, not of the evidence
        rec["projection_distorted"] = rec["pca_neighbour_agreement"] < thr_j
        rec["reasons"] = reasons
        out[c] = rec
    return out


# ------------------------------------------------------------------ votes, structured features
def votes_maps() -> dict:
    out = {}
    for term in ["2018-2022", "2022-2026"]:
        e = votes_mod.agreement_embedding(term)
        out[term] = e
    return out


def structured_features(st: State) -> dict:
    vs = json.loads((OUT / "votes_summary.json").read_text())
    affs = json.loads((OUT / "affiliations_seed.json").read_text()) if (OUT / "affiliations_seed.json").exists() else []
    claims = [json.loads(l) for l in (OUT / "claims.jsonl").open()]
    by_c = defaultdict(list)
    for c in claims:
        if c["candidate_id"]:
            by_c[c["candidate_id"]].append(c)
    sources = {s["source_id"]: s for s in json.loads((OUT / "sources.json").read_text())}
    bvj = json.loads((OUT / "board_votes.json").read_text()) if (OUT / "board_votes.json").exists() else {}
    bv = bvj.get("per_candidate", {})
    bv_meet = len(bvj.get("meetings", []))
    feats = {}
    for cand in st.cands:
        cid = cand["candidate_id"]
        cl = by_c.get(cid, [])
        # years in elected office: only from documented terms (affiliations with kind elected_office)
        terms = [a for a in affs if a.get("candidate_id") == cid and a.get("kind") == "elected_office"]
        years, basis = None, None
        spans = []
        for a in terms:
            try:
                s = int(str(a.get("start"))[:4])
                e = int(str(a.get("end"))[:4]) if a.get("end") else 2026
                spans.append((s, e))
            except (TypeError, ValueError):
                continue
        if spans:
            covered = set()
            for s, e in spans:
                covered |= set(range(s, max(e, s + 1)))
            years, basis = len(covered), "documented terms (affiliation records)"
        if cid in vs["summary"]:
            r = vs["summary"][cid]
            yrs = (np.datetime64(r["last_vote"]) - np.datetime64(r["first_vote"])).astype(int) / 365.25
            if years is None or yrs > years:
                years, basis = round(float(yrs), 1), "span of recorded Council votes (open data)"
        mgmt = [c for c in cl if c["management_signal"]]
        csrc = {c["source_id"] for c in cl}
        feats[cid] = dict(
            years_elected_office=years, years_basis=basis or "no elected office documented in collected sources",
            management_responsibility=("documented" if mgmt else None),
            management_evidence=[m["claim_id"] for m in mgmt[:5]],
            vote_record=("available" if cid in vs["summary"] else
                         "board_minutes_partial" if cid in bv else
                         "not_collected_board_minutes" if cand["race"] in ("park", "school") and terms else
                         "none_no_council_seat"),
            vote_record_note=("City Council electronic votes (open data)" if cid in vs["summary"] else
                              f"Board votes parsed from {bv_meet} minutes PDF(s) in the registry - partial coverage"
                              if cid in bv else
                              "Board vote records exist only in meeting minutes; none parsed for this member"
                              if cand["race"] in ("park", "school") and terms else
                              "No City Council voting record: not a Council member 2016-2026"),
            n_sources=len(csrc),
            n_independent_sources=sum(1 for s in csrc if sources.get(s, {}).get("source_type") in
                                      ("news", "official_record", "employer_bio", "questionnaire", "advocacy_assessment")),
            n_claims=len(cl), n_canonical_claims=sum(c["dedup_canonical"] for c in cl))
    return feats


# ------------------------------------------------------------------ main
def build(sensitivity: bool = True) -> dict:
    st = State()
    SITE.mkdir(parents=True, exist_ok=True)
    result = dict(generated_at=now_iso(), races={}, cross_race={})
    for race in RACES:
        r = {}
        for view, mode in [("policy", "individual"), ("policy", "with_platform"), ("platform", "individual"),
                           ("experience", "individual"), ("actions", "individual")]:
            key = f"{view}:{mode}"
            print(f"  {race} {key}", flush=True)
            r[key] = analyse_map(st, race, view, mode, sensitivity=sensitivity and view != "platform")
        result["races"][race] = r
    print("  cross-race policy", flush=True)
    result["cross_race"]["policy:individual"] = analyse_map(st, None, "policy", "individual", sensitivity=False)
    result["cross_race"]["warning"] = ("Cross-race map: mayor, council, Park Board and school trustee mandates "
                                       "differ. Similar language does not imply similar responsibilities.")
    result["votes"] = votes_maps()
    result["features"] = structured_features(st)
    (SITE / "analysis.json").write_text(json.dumps(result, ensure_ascii=False))
    return {r: {k: v["n"] for k, v in result["races"][r].items()} for r in RACES}


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
