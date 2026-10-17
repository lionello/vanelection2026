"""Candidate profiles (standardised evidence summaries), site data, exports and
the methods report."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict

import markdown
import numpy as np
import yaml

from .embed import config, CFG
from .fetch import ROOT, now_iso, ACCESS_LOG

OUT = ROOT / "data" / "processed"
SITE = ROOT / "data" / "site"
EXP = ROOT / "data" / "exports"

RANK_UNC = {"low": 0, "medium": 1, "medium-high": 2, "high": 3}
TOPIC_LABELS = {
    "housing": "Housing supply & approvals", "displacement": "Displacement & renter protections",
    "coops_public_housing": "Co-ops, social & public housing", "homelessness": "Homelessness, shelters & supportive housing",
    "drug_policy": "Drug policy & mental health", "policing": "Policing & public safety",
    "active_transport": "Walking & cycling safety", "transit": "Transit", "taxes": "Taxes & budget",
    "public_services": "Public services", "parks": "Parks & recreation", "education": "Education & schools",
    "governance": "Governance & accountability"}
SLOTS = dict(policy_per_topic=2, experience=5, actions=4, commentary=2, assessments=4)


def rank(c):
    return (not c["dedup_canonical"], RANK_UNC.get(c["uncertainty"], 3),
            c["verification_status"] == "commentary_secondary", -min(len(c["claim"]), 280))


def cite(c, sources):
    s = sources.get(c["source_id"], {})
    return dict(claim_id=c["claim_id"], claim=c["claim"][:320], evidence_type=c["evidence_type"],
                verification_status=c["verification_status"], uncertainty=c["uncertainty"],
                topic=c["topic"], date=c["date"], source_id=c["source_id"], passage_id=c["passage_id"],
                page=c.get("page"), publisher=s.get("publisher"), source_type=s.get("source_type"),
                url=s.get("url"), via=(s.get("fetch") or {}).get("via"), housing_forms=c["housing_forms"],
                perspective=s.get("perspective"), quoted_document=c.get("quoted_document"),
                attribution=c["attribution"])


def profiles(cands, claims, sources, feats, platform_topics_by_party, cfg):
    by_c = defaultdict(list)
    for c in claims:
        if c["candidate_id"]:
            by_c[c["candidate_id"]].append(c)
    out = {}
    for cand in cands:
        cid = cand["candidate_id"]
        cl = by_c.get(cid, [])
        pol = {}
        for t in cfg["topics"]:
            tc = sorted([c for c in cl if c["evidence_type"] == "individual_position" and t in c["topics"]], key=rank)
            if tc:
                pol[t] = dict(status="position_found", items=[cite(c, sources) for c in tc[:SLOTS["policy_per_topic"]]],
                              n_total=len(tc))
            elif t in platform_topics_by_party.get(cand["party_id"], set()):
                pol[t] = dict(status="party_platform_only", items=[], n_total=0,
                              note="No individual statement found; the party platform addresses this topic.")
            else:
                pol[t] = dict(status="not_found", items=[], n_total=0,
                              note="Not found in collected sources (absence of evidence, not a position).")
        exp = sorted([c for c in cl if c["evidence_type"] in ("elected_experience", "professional_experience", "education")],
                     key=lambda c: ({"elected_experience": 0, "professional_experience": 1, "education": 2}[c["evidence_type"]], rank(c)))
        act = sorted([c for c in cl if c["evidence_type"] in ("reported_action", "claimed_outcome")],
                     key=lambda c: (c["evidence_type"] == "claimed_outcome", rank(c)))
        com = sorted([c for c in cl if c["evidence_type"] == "commentary"], key=rank)
        shared = sorted([c for c in cl if c["evidence_type"] == "shared_campaign_language"], key=rank)
        ass = sorted([c for c in cl if c["evidence_type"] == "third_party_assessment"], key=rank)
        # diversify experience slots across sources
        def diverse(items, n):
            picked, seen = [], set()
            for c in items:
                if c["dedup_group"] in seen:
                    continue
                if len([p for p in picked if p["source_id"] == c["source_id"]]) >= 2:
                    continue
                picked.append(c)
                seen.add(c["dedup_group"])
                if len(picked) >= n:
                    break
            return picked
        out[cid] = dict(
            policy=pol,
            experience=[cite(c, sources) for c in diverse(exp, SLOTS["experience"])],
            actions=[cite(c, sources) for c in diverse(act, SLOTS["actions"])],
            commentary=[cite(c, sources) for c in diverse(com, SLOTS["commentary"])],
            assessments=[cite(c, sources) for c in diverse(ass, SLOTS["assessments"])],
            shared_language=[cite(c, sources) | {"shared_with": c.get("shared_with", [])} for c in diverse(shared, 3)],
            counts=dict(Counter(c["evidence_type"] for c in cl)),
            coverage=dict(topics_with_positions=sum(1 for v in pol.values() if v["status"] == "position_found"),
                          topics_platform_only=sum(1 for v in pol.values() if v["status"] == "party_platform_only"),
                          topics_not_found=sum(1 for v in pol.values() if v["status"] == "not_found")),
            features=feats.get(cid, {}))
    return out


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def build() -> dict:
    cfg = config()
    cands = json.loads((OUT / "candidates.json").read_text())
    ballot_meta = json.loads((OUT / "ballot_meta.json").read_text())
    sources_l = json.loads((OUT / "sources.json").read_text())
    sources = {s["source_id"]: s for s in sources_l}
    claims = [json.loads(l) for l in (OUT / "claims.jsonl").open()]
    analysis = json.loads((SITE / "analysis.json").read_text())
    affs = json.loads((OUT / "affiliations_seed.json").read_text()) if (OUT / "affiliations_seed.json").exists() else []
    gaps = json.loads((OUT / "gaps_seed.json").read_text()) if (OUT / "gaps_seed.json").exists() else []
    votes = json.loads((OUT / "votes_summary.json").read_text())
    board = json.loads((OUT / "board_votes.json").read_text()) if (OUT / "board_votes.json").exists() else {"motions": [], "per_candidate": {}}
    emb_info = json.loads((OUT / "embedding_info.json").read_text())
    feats = analysis["features"]
    platform_topics = defaultdict(set)
    for c in claims:
        if c["candidate_id"] is None and c["party_id"] and c["evidence_type"] == "party_commitment" and c["dedup_canonical"]:
            for t in c["topics"]:
                platform_topics[c["party_id"]].add(t)
    prof = profiles(cands, claims, sources, feats, platform_topics, cfg)
    # affiliations attach
    for c in cands:
        c["affiliations"] = [a for a in affs if a.get("candidate_id") == c["candidate_id"]]
        c["gaps"] = [g for g in gaps if g.get("candidate_id") == c["candidate_id"]]
        c["profile"] = prof[c["candidate_id"]]
        c["vote_summary"] = votes["summary"].get(c["candidate_id"])
        dep = votes.get("abc_caucus_departures", {})
        c["caucus_departures"] = (dict(n=len(dep["departures"].get(c["candidate_id"], [])),
                                       of=dep["contested_votes_as_caucus_member"][c["candidate_id"]],
                                       items=dep["departures"].get(c["candidate_id"], []), method=dep["method"],
                                       sources=dep["sources"])
                                  if c["candidate_id"] in dep.get("contested_votes_as_caucus_member", {}) else None)
        bm = [m for m in board.get("motions", []) if c["candidate_id"] in m.get("candidate_votes", {})]
        c["board_votes"] = dict(counts=board["per_candidate"].get(c["candidate_id"]), note=board.get("note"),
                                motions=[dict(date=m["date"], board=m["board"], result=m["result"], text=m["text"],
                                              vote=m["candidate_votes"][c["candidate_id"]], is_amendment=m["is_amendment"],
                                              source_id=m["source_id"]) for m in bm]) if bm else None
        c["person_id"] = c["display_name"].lower().replace(" ", "-")
    persons = Counter(c["person_id"] for c in cands)
    for c in cands:
        c["also_on_ballot_as"] = [x["candidate_id"] for x in cands if x["person_id"] == c["person_id"]
                                  and x["candidate_id"] != c["candidate_id"]] if persons[c["person_id"]] > 1 else []
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "candidates.json").write_text(json.dumps(cands, ensure_ascii=False))
    slim_sources = [{k: s.get(k) for k in ("source_id", "url", "title", "publisher", "published", "source_type",
                                           "candidates", "party", "perspective", "availability", "duplicate_of",
                                           "n_passages", "notes", "content_hash")} | {"via": (s.get("fetch") or {}).get("via"),
                                           "wayback_ts": (s.get("fetch") or {}).get("wayback_ts"),
                                           "retrieved_at": (s.get("fetch") or {}).get("retrieved_at")}
                    for s in sources_l]
    (SITE / "sources.json").write_text(json.dumps(slim_sources, ensure_ascii=False))
    cdir = SITE / "claims"
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir()
    by = defaultdict(list)
    for c in claims:
        by[c["candidate_id"] or f"party-{c['party_id']}"].append(c)
    for k, v in by.items():
        (cdir / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False))
    access = [json.loads(l) for l in ACCESS_LOG.open()] if ACCESS_LOG.exists() else []
    last = {}
    for a in access:
        last[a["url"]] = a
    blocked = [a for a in last.values() if not a["outcome"].startswith("ok")]
    coverage = dict(
        candidates_total=len(cands),
        with_any_claim=sum(1 for c in cands if c["profile"]["counts"]),
        with_policy_position=sum(1 for c in cands if c["profile"]["coverage"]["topics_with_positions"] > 0),
        with_experience=sum(1 for c in cands if c["profile"]["experience"]),
        with_vote_record=sum(1 for c in cands if c["vote_summary"]),
        without_any_evidence=[c["candidate_id"] for c in cands if not c["profile"]["counts"]],
    )
    meta = dict(generated_at=now_iso(), ballot=ballot_meta, embedding=emb_info, config=cfg,
                config_sha256=hashlib.sha256(CFG.read_bytes()).hexdigest(),
                counts=dict(sources=len(sources_l), sources_ok=sum(s.get("availability") == "ok" for s in sources_l),
                            claims=len(claims), canonical_claims=sum(c["dedup_canonical"] for c in claims),
                            chunks=sum(1 for _ in (OUT / "chunks.jsonl").open())),
                coverage=coverage, votes_dataset=votes["dataset"], vote_value_semantics=votes["value_semantics"],
                inaccessible_sources=blocked, topic_labels=TOPIC_LABELS,
                views={k: v["label"] for k, v in cfg["views"].items()},
                comprehensive=False,
                comprehensiveness_note=("All 130 ballot entries are accounted for in the candidate list, but "
                                        "evidence coverage is partial: see coverage. Council ballot entries rest "
                                        "on a secondary source for ballot numbers only; names are corroborated by Elections BC."))
    (SITE / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    exports(cands, claims, sources_l, analysis, meta)
    return dict(coverage=coverage, counts=meta["counts"])


def exports(cands, claims, sources_l, analysis, meta):
    EXP.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cands:
        f = c["profile"]["features"]
        rows.append(dict(candidate_id=c["candidate_id"], race=c["race"], ballot_number=c["ballot_number"],
                         name=c["display_name"], name_cjk=c["name_cjk"], party=c["party"],
                         party_label_on_ballot=c["party_label_on_ballot"], ballot_verification=c["ballot_verification"],
                         ballot_source=c["ballot_source"]["url"], official_candidate_page=c["official_candidate_page"],
                         years_elected_office=f.get("years_elected_office"), years_basis=f.get("years_basis"),
                         management_responsibility=f.get("management_responsibility"), vote_record=f.get("vote_record"),
                         n_sources=f.get("n_sources"), n_claims=f.get("n_claims"),
                         topics_with_positions=c["profile"]["coverage"]["topics_with_positions"],
                         campaign_links=[l["url"] for l in c["campaign_links"]],
                         official_documents=[d["url"] for d in c["official_documents"]],
                         affiliations=[f"{a.get('organization')} ({a.get('role')}, {a.get('start')}-{a.get('end') or 'present'})"
                                       for a in c["affiliations"]]))
    write_csv(EXP / "candidates.csv", rows, list(rows[0]))
    (EXP / "candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=1))
    cf = ["claim_id", "candidate_id", "party_id", "topic", "topics", "claim", "evidence_type", "source_id", "passage_id",
          "page", "supporting_passage", "date", "date_basis", "verification_status", "uncertainty", "attribution",
          "housing_forms", "dedup_group", "dedup_canonical", "also_in_sources"]
    write_csv(EXP / "claims.csv", claims, cf)
    shutil.copy(OUT / "claims.jsonl", EXP / "claims.jsonl")
    sf = ["source_id", "url", "title", "publisher", "published", "source_type", "candidates", "party", "perspective",
          "availability", "duplicate_of", "content_hash", "n_passages", "notes"]
    write_csv(EXP / "sources.csv", [s | {"via": (s.get("fetch") or {}).get("via")} for s in sources_l], sf + ["via"])
    shutil.copy(OUT / "votes_candidates.csv", EXP / "council_votes_candidates.csv")
    vs = json.loads((OUT / "votes_summary.json").read_text())
    write_csv(EXP / "council_vote_agreement_pairs.csv", vs["pairs"], list(vs["pairs"][0]))
    if (OUT / "board_votes.json").exists():
        shutil.copy(OUT / "board_votes.json", EXP / "board_votes_parsed_minutes.json")
    shutil.copy(OUT / "chunk_vectors.npy", EXP / "chunk_embeddings.npy")
    write_csv(EXP / "chunk_embeddings_index.csv",
              [json.loads(l) | {"row": i} for i, l in enumerate((OUT / "chunks.jsonl").open())],
              ["row", "chunk_id", "subject", "view", "topic", "source_id", "passage_id", "near_duplicate_of", "text"])
    for v in ["policy", "platform", "experience", "actions"]:
        if (OUT / f"repr_{v}.npz").exists():
            shutil.copy(OUT / f"repr_{v}.npz", EXP / f"topic_vectors_{v}.npz")
    shutil.copy(CFG, EXP / "analysis_config.yaml")
    (EXP / "embedding_model.json").write_text(json.dumps(meta["embedding"], indent=1))
    (EXP / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False))
    if ACCESS_LOG.exists():
        shutil.copy(ACCESS_LOG, EXP / "access_log.jsonl")
    md = methods_report(meta, analysis, cands)
    (EXP / "methods_report.md").write_text(md)
    html = markdown.markdown(md, extensions=["tables", "toc"])
    (EXP / "methods_report.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Methods report</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:860px;margin:2rem auto;"
        "padding:0 16px;color:#1d1d1f;background:#fff}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;"
        "padding:4px 8px}code{background:#f3f3f3;padding:0 3px}@media(prefers-color-scheme:dark){body{background:#151515;"
        "color:#e8e8e8}code{background:#2a2a2a}td,th{border-color:#444}}</style>" + html)
    with zipfile.ZipFile(EXP / "vancouver-2026-explorer-export.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in EXP.iterdir():
            if p.suffix != ".zip":
                z.write(p, p.name)


def fmt_ev(m):
    ev = (m.get("pca") or {}).get("explained_variance_ratio") or []
    return f"{round(100 * sum(ev[:2]))}%" if len(ev) >= 2 else "n/a"


def methods_report(meta, analysis, cands) -> str:
    b = meta["ballot"]
    e = meta["embedding"]
    cov = meta["coverage"]
    cfg = meta["config"]
    lines = [
        "# Vancouver 2026 candidate explorer: methods report", "",
        f"Generated {meta['generated_at']}. Election day 2026-10-17. **All data is pre-election**; no results are included.", "",
        "## 1. Ballot", "",
        f"Candidates: **{b['total']}** (mayor {b['counts']['mayor']}, council {b['counts']['council']}, "
        f"Park Board {b['counts']['park']}, school trustee {b['counts']['school']}).", "",
        "* vancouver.ca returns a Cloudflare bot challenge (HTTP 403) to automated clients. It was not circumvented.",
        "* Mayor, Park Board and school trustee lists come from Internet Archive captures of the City's own "
        "candidate pages (snapshot timestamps are stored with every record).",
        "* The councillor page and the City's candidate-list PDF have no usable capture. Council ballot numbers come from "
        "Daily Hive's ballot-order article (2026-09-22), which matches the official pages for all 74 mayor, Park Board "
        "and school entries (name and ballot number).",
        f"* Every ballot entry ({b['elections_bc']['matched']}/{b['total']}) was matched to Elections BC's official "
        "registered-candidates list (name, office, affiliation; two documented name variants). Council entries are "
        "therefore `official_registration_plus_secondary_ballot_number`. Registered but not on the ballot: "
        + ", ".join(f"{r['name']} ({r['affiliation']})" for r in b['elections_bc']['registered_not_on_ballot']) + ".",
        "* Candidate IDs are `<race>-<name-slug>`; one person on two ballots (Naomi Chocyk: council and school) has two IDs "
        "linked by `also_on_ballot_as`.", "",
        "## 2. Evidence collection", "",
        f"Sources registered: {meta['counts']['sources']} ({meta['counts']['sources_ok']} with extractable text). "
        "Each source keeps its original bytes, extracted text, URL, publisher, publication date when stated, retrieval "
        "time, SHA-256 hash and the route used (direct or Internet Archive). Inaccessible URLs are logged "
        f"({len(meta['inaccessible_sources'])} at build time).", "",
        "Source discovery was done by research agents that recorded URLs only; every fact shown in the app is a verbatim "
        "sentence from a cached source.", "",
        "## 3. Claims", "",
        f"Claims: {meta['counts']['claims']} ({meta['counts']['canonical_claims']} canonical after de-duplication). "
        "Each claim is one verbatim sentence with `candidate_id` (or `party_id` for platforms), topic(s), evidence type, "
        "source and passage id, date, verification status and uncertainty.", "",
        "Evidence types: individual_position, party_commitment, elected_experience, professional_experience, education, "
        "reported_action, claimed_outcome (a campaign's own claim of a result, never treated as verified), commentary.", "",
        "Classification is rule-based (keyword patterns published in `vanelect/claims.py`), so every label can be audited. "
        "It is not perfect: misclassified sentences are expected, and the uncertainty field and source links let readers check.", "",
        "Attribution: a sentence is assigned to a candidate when it is on that candidate's own page, under a heading that "
        "names them, or when it names them. Party platform text is attributed to the party only and is never added to an "
        "individual's personal record.", "",
        "Missing values are distinct: `not_found` (no statement in collected sources), `party_platform_only`, "
        "`none_no_council_seat` (no voting record because they never sat on Council), and per-vote values Absent, "
        "Abstain, No Vote, Declared Conflict (recusal) and Ineligible. None of those is a vote against.", "",
        "Housing forms are tagged separately and never merged: SRO, self-contained micro-unit, supportive housing, "
        "shelter, co-op, social housing, market rental.", "",
        "## 4. Representations", "",
        f"Embedding model: `{e.get('model')}` ({e.get('backend')}, revision `{e.get('revision', 'n/a')}`, "
        f"{e.get('dimensions')} dimensions, max {e.get('max_seq_length')} tokens). Preprocessing: {e.get('preprocessing')}. "
        "Every chunk vector is L2-normalised.", "",
        "Views are kept separate: (1) individual policy positions, (2) party platform, embedded once per party and linked to "
        "candidates, (3) experience and governance, (4) reported actions and campaign-claimed outcomes. Council recorded "
        "votes form a separate agreement-based map.", "",
        "Aggregation: chunks are de-duplicated (exact sentence duplicates per subject, then cosine "
        f">= {cfg['dedup']['near_duplicate_cosine']}). Chunks are averaged per topic and re-normalised. Topic vectors are then "
        "averaged with equal (or user-chosen) weights over the topics that have evidence, so a prolific campaign "
        "does not outweigh a terse one. Missing topics are excluded, never zero-filled.", "",
        "\"Include party platform\" mode: for each topic, the candidate's vector and the party's are summed and "
        "re-normalised. Topics only the platform covers are filled from the platform and marked.", "",
        f"Insufficient evidence: fewer than {cfg['thresholds']['min_chunks']} chunks or fewer than "
        f"{cfg['thresholds']['min_topics']} topics (policy and actions views). Such candidates are listed, not plotted.", "",
        "## 5. Analysis", "",
        "* PCA is the primary projection. Explained variance is shown with each map.",
        f"* UMAP is optional: n_neighbors {cfg['umap']['n_neighbors']} (capped at n-1), min_dist {cfg['umap']['min_dist']}, "
        f"cosine metric, seed {cfg['umap']['seeds'][0]}. Its axes have no inherent meaning.",
        "* Nearest neighbours are computed by cosine in the original embedding space, each with its most similar "
        "supporting passage pairs.",
        "* Each race is analysed separately. A cross-race policy-only map is offered with a warning that mandates differ.",
        "* No axis is labelled left/right or competent/incompetent, and no supervised model is trained.", "",
        "Sensitivity (per candidate):",
        f"* source dropout: {cfg['sensitivity']['source_dropout']['runs']} runs, each source dropped with p="
        f"{cfg['sensitivity']['source_dropout']['drop_probability']}; Procrustes-aligned PCA displacement and neighbour Jaccard;",
        f"* topic weights: {cfg['sensitivity']['topic_weights']['runs']} Dirichlet(alpha={cfg['sensitivity']['topic_weights']['dirichlet_alpha']}) re-weightings;",
        "* party-platform toggle: neighbour Jaccard between modes;",
        f"* UMAP seeds {cfg['umap']['seeds']};",
        "* projection fidelity: 2-D neighbours compared with original-space neighbours.",
        f"A placement is flagged unstable when median displacement > {cfg['sensitivity']['unstable_displacement']} or "
        f"neighbour Jaccard < {cfg['sensitivity']['unstable_neighbour_jaccard']}.", "",
        "### Results of the sensitivity tests (this build)", "",
        "| race | view | plotted | PC1+PC2 variance | unstable | 2-D distorted | same-party top-3 neighbours |",
        "|---|---|---|---|---|---|---|"] + [
        f"| {race} | {k} | {m['n']} | {fmt_ev(m)} | {sum(x['unstable'] for x in m.get('stability', {}).values())} | "
        f"{sum(x.get('projection_distorted', False) for x in m.get('stability', {}).values())} | "
        f"{'' if m.get('same_party_neighbour_rate') is None else str(round(100 * m['same_party_neighbour_rate'])) + '%'} |"
        for race, maps in analysis["races"].items() for k, m in maps.items() if k != "platform:individual"] + ["",
        "Reading: with individual evidence only, most nearest neighbours cross party lines. Adding the shared party "
        "platform pulls each candidate toward their party, so same-party neighbour rates rise sharply. Verbatim text "
        "shared by several candidates of one party is retyped `shared_campaign_language` and kept out of individual "
        "vectors (cross-candidate cosine >= the near-duplicate threshold is treated the same way).", "",
        "## 6. Council votes", "",
        f"Source: {meta['votes_dataset']['url']} (modified {meta['votes_dataset']['modified']}, sha256 "
        f"`{meta['votes_dataset']['sha256'][:16]}...`). The City says minutes remain the official record. Motions are "
        f"matched by (meeting_id, vote_number). Agreement is computed only on matching contested motions (not carried "
        f"unanimously) where both members cast For/Against, requires >= {cfg['votes']['min_overlap']} shared votes, and is "
        "never compared across different councils (2018-22 and 2022-26 are separate maps).", "",
        "Park Board and School Board votes are not published as open data. Minutes PDFs in the source registry are "
        "parsed (`vanelect/board_votes.py`) for motions, results and named In favour / Opposed / Absent / Abstained "
        "lists and conflict declarations. Coverage is partial (only the meetings collected), and no agreement map is "
        "built from them.", "",
        "## 7. Coverage and limitations", "",
        f"* Candidates with any evidence: {cov['with_any_claim']}/{cov['candidates_total']}; with at least one individual "
        f"policy position: {cov['with_policy_position']}; with experience evidence: {cov['with_experience']}; "
        f"with Council vote records: {cov['with_vote_record']}.",
        f"* Candidates with no evidence collected: {len(cov['without_any_evidence'])}.",
        "* Coverage is NOT comprehensive. Absence of a statement is shown as `not_found`, not as a position.",
        "* No language model wrote or paraphrased any claim. Classification uses rules, and embeddings come from one "
        "local model.",
        "* Commentary (e.g. Allen Pike's guide) is shown as labelled secondary context and is excluded from the "
        "embeddings.", "",
        "## 8. Reproduce", "", "```", "make setup   # once", "make build   # fetch, extract, claims, embed, analyse, export",
        "make serve", "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
