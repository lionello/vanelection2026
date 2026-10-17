"""Atomic claim extraction with transparent, rule-based classification.

No language model is used to write or paraphrase claims: every claim's text is
a verbatim sentence from a cached source, with its passage id. Classification
(topic, evidence type) uses the published keyword rules below so that every
label can be audited and reproduced.

Claim schema (data/processed/claims.jsonl):
  claim_id, candidate_id | party_id, topic, topics, claim, evidence_type,
  source_id, passage_id, supporting_passage, date, date_basis,
  verification_status, uncertainty, attribution, housing_forms, dedup_group,
  also_in_sources
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict

from .fetch import ROOT

OUT = ROOT / "data" / "processed"

# ------------------------------------------------------------------ topic lexicon
TOPICS = {
    "housing": r"\bhousing\b|\bhomes?\b|\brent(al|als)?\b|\bzoning\b|\bdensity\b|\bdevelop(ment|er|ers)\b|\bpermit(s|ting)?\b|\btownhouse|\bmultiplex|\bafford(able|ability)\b|\bapartment|\bbuilding approvals?|\bupzon|\bmissing middle",
    "displacement": r"\bdisplace(ment|d)?\b|\brenovict|\btenant(s)?\b|\brenter(s)?\b|\beviction|\btenant protection|\bright of first refusal|\brent control|\bvacancy control|\bTRAP\b|\brelocation",
    "coops_public_housing": r"\bco-?op(erative)?s?\b(?! students)|\bsocial housing\b|\bpublic housing\b|\bnon-?profit housing|\bcommunity housing\b|\bhousing authority|\bBC Housing\b|\bnon-market\b|\bcity-owned land|\bland trust",
    "homelessness": r"\bhomeless(ness)?\b|\bunhoused\b|\bshelters?\b|\bsupportive housing\b|\bencampment|\bSROs?\b|\bsingle room occupancy|\bDowntown Eastside\b|\bDTES\b|\btent(s)?\b|\bmodular housing|\boutreach",
    "drug_policy": r"\bdrugs?\b|\bdecrimin|\bsafe supply\b|\boverdose|\bharm reduction\b|\binvoluntary care\b|\bsafe consumption|\bsupervised consumption|\btreatment\b|\brecovery\b|\baddiction|\bsubstance use|\bopioid|\bfentanyl|\bmental health",
    "policing": r"\bpolic(e|ing)\b|\bVPD\b|\bofficers?\b|\bcrime\b|\bpublic safety\b|\bsafety\b(?! (of|for) (cyclists|pedestrians))|\bstreet disorder|\bpolice board|\bcar 87|\bcommunity safety|\bSLO\b|\bschool liaison",
    "active_transport": r"\bbike|\bcycl(ing|ists?|e)\b|\bpedestrian|\bwalk(ing|able|ability)?\b|\bsidewalk|\bcrosswalk|\bvision zero\b|\btraffic safety|\bspeed limit|\bbike lanes?|\bgreenway|\bcar-free|\bplaza",
    "transit": r"\btransit\b|\bTransLink\b|\bSkyTrain\b|\bbus(es)?\b|\bUBC (line|extension)|\bBroadway (subway|plan)|\bMillennium Line|\brapid transit|\bMayors'? Council",
    "taxes": r"\btax(es|ation)?\b|\bbudget\b|\bproperty tax|\bspending\b|\bfiscal|\bdebt\b|\bborrowing\b|\bdevelopment cost|\bDCLs?\b|\bCACs?\b|\bfees?\b|\bwaste(ful)?\b|\baudit\b|\bauditor general",
    "public_services": r"\blibrar(y|ies)\b|\bchild ?care\b|\bcommunity centres?\b|\bpools?\b|\bfire\b|\bsanitation|\bgarbage|\bstreet cleaning|\bwater\b|\bsewer|\bpermitting times?|\b3-?1-?1\b|\bcity services\b|\bseniors?\b|\baccessib",
    "parks": r"\bparks?\b|\bPark Board\b|\bgreen space|\btrees?\b|\bbeach(es)?\b|\bplayground|\bplayfields?|\brecreation\b|\bsports? fields?|\bStanley Park\b|\bgolf\b|\bseawall",
    "education": r"\bschools?\b|\bstudents?\b|\bteachers?\b|\beducation\b|\bVSB\b|\bschool board\b|\btrustees?\b|\bclassrooms?\b|\bseismic\b|\bcurriculum|\bK-?12\b|\blearning\b",
    "governance": r"\btransparen|\baccountab|\bgovernance\b|\bethics\b|\bconflict of interest|\bconsultation\b|\bcity hall\b|\bbureaucra|\bwards?\b|\bcampaign finance|\blobby|\bdemocra|\bpartisan|\babolish|\bindependen(t|ce) of|\bcouncil (meeting|procedure)",
}
TOPIC_RE = {k: re.compile(v, re.I) for k, v in TOPICS.items()}

# Distinct housing forms are tagged separately and never merged.
HOUSING_FORMS = {
    "sro": r"\bSROs?\b|\bsingle room occupancy|\bSRA\b|\bsingle room accommodation",
    "micro_unit": r"\bmicro-?(suites?|units?|homes?)\b|\bself-contained micro|\bmicro-?dwelling",
    "supportive_housing": r"\bsupportive housing\b|\bsupportive homes\b|\btemporary modular housing|\bmodular supportive",
    "shelter": r"\bshelters?\b|\bwarming centres?|\bnavigation cent",
    "coop": r"\bco-?op(erative)? (housing|homes)|\bhousing co-?op",
    "social_housing": r"\bsocial housing\b|\bnon-market housing\b|\bpublic housing\b",
    "market_rental": r"\bmarket rental|\bpurpose-built rental|\bsecured rental",
}
HF_RE = {k: re.compile(v, re.I) for k, v in HOUSING_FORMS.items()}

# ------------------------------------------------------------------ evidence-type cues
ELECTED = re.compile(r"\b(elected|re-elected|served|serves|sworn in|appointed|has chaired|chaired|chairs)\b.*\b(council|councillor|commissioner|trustee|mayor|park board|school board)\b|\b(councillor|commissioner|trustee|mayor)\b.*\b(since|from|elected in|in) (19|20)\d\d\b.*\b(elected|served|term)|\bas (a |an )?(city )?(councillor|commissioner|trustee|mayor)\b.*\b(chaired|served|led|sat on)\b|\bchair(ed)? (of )?the (park|school|police|finance)|\bvice-chair", re.I)
PROF_INTRO = re.compile(r"^As an? [\w\s'-]{3,50}, (she|he|they|[A-Z]\w+) (has|have|had|was|is|helped|led|worked|spent)\b")
SIGNED = re.compile(r"[”\"]\s*[–—-]\s*[A-Z][a-z]+")
EXPERIENCE = re.compile(r"\b(years? (of )?experience|career|chaired|served on|worked (as|at|for|in)|works (as|at|for)|served as|serves as|founder|co-?founded|founded|director|manager|executive|CEO|president|owner|lawyer|engineer|nurse|teacher|professor|consultant|employed|board (member|of directors)|volunteer(ed|s)? (with|at|for)|chair of|led (a|the)|managed|oversaw|responsible for|budget of|staff of|team of)\b", re.I)
EDUCATION = re.compile(r"\b(degree|bachelor|master'?s|MBA|PhD|doctorate|diploma|graduate[d]? (of|from)|studied|certified|certification|designation|CPA|P\.Eng|J\.D\.|LL\.?B|university of|UBC|SFU|BCIT|Langara)\b", re.I)
MGMT = re.compile(r"\b(budget of|\$[\d.,]+ ?(million|billion|m|b)\b.*\b(budget|portfolio|operation)|managed (a )?(team|staff|budget)|staff of \d+|team of \d+|\d+ (employees|staff)|oversaw|responsible for (a|the) .*(budget|team|staff|program))", re.I)
OUTCOME = re.compile(r"\b(I|we|he|she|they|[A-Z][a-z]+) (have |has )?(delivered|secured|built|reduced|increased|achieved|passed|won|saved|created|launched|opened|completed|fixed|cut|froze|hired|added|got)\b", re.I)
POSITION = re.compile(r"\b(will|would|must|should|ought|need(s|ed)? to|needs?\b|cannot|can't|deserves?|plan(s)? to|commit(s|ted)? to|propose[sd]?|support(s|ing)?|oppose[sd]?|opposing|believe[sd]?|priorit(y|ies|ize|ise)|fight(s|ing)? for|push(es|ing)? for|advocat(e|es|ing) for|stand(s)? (for|against)|want(s)?\b|calls? for|pledge[sd]?|promise[sd]?|vision|time to|important to|let's|we'll|i'll|end the|stop the|restore|protect|expand|reduce|increase|cut|freeze|ban|scrap|reverse|invest in|keep)\b", re.I)
SPEECH = re.compile(r"[\"“”‘’'].{8,}[\"“”‘’'].*\b(said|says|told|wrote|stated|argued|added)\b|\b(said|says|told|wrote|stated|argued|added)\b.*[\"“”].{8,}", re.I)
QUOTE = re.compile(r"[\"“”]|\b(said|says|told|according to)\b", re.I)
BOILER = re.compile(r"\b(donate|volunteer sign|sign up|subscribe|newsletter|sponsored|is running for (city )?council|on the ballot for|candidate \d+ on the ballot|contribute this|we acknowledge|unceded|pdf file|retrieved \d|archived from|accessed \d|cookie|privacy policy|all rights reserved|follow us|share this|click here|read more|copyright|paid for by|authorized by|financial agent|log ?in|menu|skip to)\b", re.I)


ATTRIB_LINE = re.compile(r"^\s*[—–]\s*(.+?)\s*,\s*(?:page\s*\d+\s*,\s*)?(?:(?:accessed|retrieved)\b|\d{4}-\d\d-\d\d)", re.I)
PARTY_DOC = re.compile(r"platform|policy|policies|fight for vancouver|blueprint|priorities|plan\b|party|abc|onecity|one city|cope|green|team\b|vote vancouver|liberal|bright future|trueblue|parks party|press release", re.I)
PARTY_SUBJECT = re.compile(r"^\W*(ABC(?: Vancouver)?|OneCity(?: Vancouver)?|COPE|(?:The )?Greens?|(?:The )?Green Party|TEAM|Vote Vancouver|(?:The )?Vancouver Liberals|Bright Future Vancouver|BFV|TrueBlue(?: Vancouver)?|(?:The )?Independent Parks Party)(?:'s)?\s+(will|would|is committed|commits|plans?|promises?|pledges?|proposes?|supports?|opposes?|wants?)\b")


def clean_sentence(s: str) -> str:
    s = s.strip().strip("\"“”").strip()
    return s


def mostly_names(s: str) -> bool:
    w = re.findall(r"[A-Za-z][\w'’-]*", s)
    if len(w) < 4:
        return True
    caps = sum(1 for x in w if x[0].isupper())
    return caps / len(w) > 0.7


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[\"“A-Z0-9•\-])", text)
    out = []
    for p in parts:
        p = clean_sentence(p.strip(" •-"))
        if p.startswith("—") or p.startswith("–") or p.endswith("?"):
            continue
        if p.count("|") >= 3:
            continue  # flattened table row, not a sentence
        if len(p.split()) >= 6 and len(p) <= 700 and not BOILER.search(p) and not mostly_names(p):
            out.append(p)
    return out


def topics_of(s: str) -> list[str]:
    hits = [(k, len(r.findall(s))) for k, r in TOPIC_RE.items()]
    hits = [h for h in hits if h[1]]
    hits.sort(key=lambda x: -x[1])
    return [h[0] for h in hits]


def housing_forms(s: str) -> list[str]:
    return [k for k, r in HF_RE.items() if r.search(s)]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s)).strip()


# ------------------------------------------------------------------ name matching
AMBIGUOUS_SURNAMES = {"smith", "lee", "brown", "kelly", "grant", "jones", "wu", "lin", "khan", "han", "bird",
                      "smart", "burns", "irwin", "kyle", "gee", "mah", "sim", "fry", "orr", "jung", "valor",
                      "bailey", "rogers", "potter", "joseph", "cameron", "chien", "zhou", "lowe", "singh"}
EXTRA_ALIASES = {
    "park-summit-wannamaker": ["Summit Ambeault-Wannamaker", "Ambeault-Wannamaker"],
    "park-t-uy-t-tanat-cease-wyss": ["T'uy't'tanat-Cease Wyss", "Cease Wyss", "Tʼuyʼtʼtanat-Cease Wyss"],
    "council-mohadeseh-gharib-p-arasi": ["Mohadeseh G.P. Arasi", "Mohadeseh Arasi", "Mohadeseh G. P. Arasi"],
    "school-isabel-ochoa-prieto": ["Isabel Ochoa"],
    "council-iona-bonamis": ["Iona Bonamis"],
    "council-rollergirl": ["ROLLERGIRL", "Rollergirl"],
    "school-christopher-jk-richardson": ["Christopher Richardson"],
    "council-parris-j-williams": ["Parris Williams"],
    "park-michael-robert-caditz": ["Michael Caditz"],
    "mayor-ken-sim": ["Mayor Sim", "Mayor Ken Sim"],
}


def alias_table(cands: list[dict]) -> dict[str, list[re.Pattern]]:
    surname_count = defaultdict(int)
    for c in cands:
        surname_count[norm(c["display_name"]).split()[-1]] += 1
    table = {}
    for c in cands:
        names = {c["display_name"]} | set(EXTRA_ALIASES.get(c["candidate_id"], []))
        pats = []
        for n in names:
            esc = re.escape(n).replace("\\'", "['’ʼ]").replace("\\ ", r"\s+")
            pats.append(re.compile(r"\b" + esc + r"\b", re.I))
        sur = c["display_name"].split()[-1]
        if surname_count[norm(sur).split()[-1]] == 1 and norm(sur) not in AMBIGUOUS_SURNAMES and len(sur) > 3:
            pats.append(re.compile(r"\b" + re.escape(sur) + r"\b"))  # case-sensitive surname
        table[c["candidate_id"]] = pats
    return table


def mentions(text: str, cid: str, table) -> bool:
    return any(p.search(text) for p in table[cid])


# ------------------------------------------------------------------ classification
def classify(sent: str, source_type: str, how: str = "") -> tuple[str, str]:
    """-> (evidence_type, verification_status)"""
    if source_type == "party_platform":
        return "party_commitment", "party_commitment"
    if source_type == "commentary":
        return "commentary", "commentary_secondary"
    camp = source_type in ("candidate_bio", "campaign_site", "party_candidate_list")
    if source_type == "questionnaire" and how == "name_in_sentence" and SIGNED.search(sent):
        return "individual_position", "questionnaire_response"
    if source_type == "advocacy_assessment" or (source_type == "questionnaire" and how == "name_in_sentence"):
        # the publisher's own words about a candidate: an assessment, not the candidate's position
        return "third_party_assessment", "third_party_assessment"
    if source_type == "questionnaire":
        if EDUCATION.search(sent) or (EXPERIENCE.search(sent) and not POSITION.search(sent)):
            return "professional_experience", "questionnaire_response"
        return "individual_position", "questionnaire_response"
    if ELECTED.search(sent):
        et = "elected_experience"
    elif EDUCATION.search(sent):
        et = "education"
    elif PROF_INTRO.search(sent) or (EXPERIENCE.search(sent) and not POSITION.search(sent)):
        et = "professional_experience"
    elif OUTCOME.search(sent):
        et = "claimed_outcome" if camp else "reported_action"
    elif source_type in ("news", "reference", "official_record", "employer_bio") and not SPEECH.search(sent):
        et = "reported_action" if source_type != "employer_bio" else "professional_experience"
    elif POSITION.search(sent) or source_type == "questionnaire":
        et = "individual_position"
    else:
        et = "biography"  # descriptive campaign text without a stance: shown, never embedded
    vs = {"candidate_bio": "attributed_campaign_claim", "campaign_site": "attributed_campaign_claim",
          "party_candidate_list": "attributed_campaign_claim", "official_record": "official_record",
          "employer_bio": "employer_record", "news": "independently_reported",
          "reference": "tertiary_reference"}.get(source_type, "unverified")
    if source_type == "news" and et == "individual_position":
        vs = "reported_quote_or_paraphrase"
    if et == "claimed_outcome":
        vs = "claimed_outcome_unverified"
    return et, vs


SINGLE_TAG_OK = {"candidate_bio", "campaign_site", "questionnaire", "party_candidate_list"}
SECTIONED = {"party_candidate_list", "questionnaire", "advocacy_assessment", "candidate_bio",
             "campaign_site", "official_record"}


def build() -> dict:
    cands = json.loads((OUT / "candidates.json").read_text())
    cmap = {c["candidate_id"]: c for c in cands}
    sources = {s["source_id"]: s for s in json.loads((OUT / "sources.json").read_text())}
    passages = [json.loads(l) for l in (OUT / "passages.jsonl").open()]
    table = alias_table(cands)
    _CANDS.clear()
    _CANDS.update(cmap)
    _FIRST.clear()
    for c in cands:
        _FIRST.setdefault(c["display_name"].split()[0], set()).add(c["candidate_id"])
    by_src = defaultdict(list)
    for p in passages:
        by_src[p["source_id"]].append(p)
    claims = []
    for sid, ps in by_src.items():
        src = sources[sid]
        if src.get("availability") != "ok":
            continue
        if all(c.startswith("*") for c in src["candidates"]) and src["candidates"]:
            continue  # official ballot lists: identity data only
        st = src["source_type"]
        named = [c for c in src["candidates"] if c in cmap]
        party = src.get("party")
        date = src.get("published") or (src.get("fetch") or {}).get("retrieved_at", "")[:10]
        date_basis = "published" if src.get("published") else "retrieved"
        if st == "party_platform":  # platform text is a party commitment, never an individual's record
            pid = party_id_of(party, cands)
            for p in ps:
                if p["kind"] != "text":
                    continue
                for s in sentences(p["text"]):
                    claims.append(mk(None, pid, s, p, src, date, date_basis, "party_page"))
            continue
        # which candidates can this source speak about?
        scope = named or ([c for c in cmap] if st in ("news", "commentary", "questionnaire",
                                                      "advocacy_assessment", "reference") else [])
        single = len(named) == 1 and st in ("candidate_bio", "campaign_site", "employer_bio")
        # Aggregator pages (e.g. VoteMate) follow quotes with "— <document>, page N, retrieved <date>".
        # Each block takes the attribution of the next attribution line; party documents become
        # party commitments; unattributed quote blocks on such pages are skipped.
        attrib, pending = {}, []
        has_attrib = any(ATTRIB_LINE.match(p["text"]) for p in ps if p["kind"] == "text")
        for p in ps:
            if p["kind"] != "text":
                continue
            m = ATTRIB_LINE.match(p["text"])
            if m:
                for pid in pending:
                    attrib[pid] = m.group(1)
                pending = []
            elif p["text"].lstrip().startswith(("\"", "“")) or pending:
                pending.append(p["passage_id"])
        section_owner, owner_level = None, 99
        for p in ps:
            if p["kind"] == "heading":
                lvl = p.get("level") or 2
                hits = [c for c in scope if mentions(p["text"], c, table)]
                if len(hits) == 1:
                    section_owner, owner_level = hits[0], lvl
                elif lvl <= owner_level:
                    section_owner, owner_level = None, 99
                continue
            quoted_doc = attrib.get(p["passage_id"])
            if has_attrib and quoted_doc is None and p["text"].lstrip().startswith(("\"", "“")):
                continue  # unattributed quote on an aggregator page: provenance unknown
            for s in sentences(p["text"]):
                owners, how = [], None
                if quoted_doc and PARTY_DOC.search(quoted_doc) and named:
                    pid = st_party(named[0], cmap)
                    claims.append(mk(None, pid, s, p, src, date, date_basis, "aggregator_quote_of_party_document",
                                     quoted_doc=quoted_doc))
                    continue
                ps_m = PARTY_SUBJECT.match(s)
                if ps_m and st != "news":
                    pid = party_id_of(ps_m.group(1), cands) or (st_party(named[0], cmap) if named else None)
                    if pid:
                        claims.append(mk(None, pid, s, p, src, date, date_basis, "party_subject_sentence"))
                        continue
                if single:
                    owners, how = named, "single_candidate_page"
                else:
                    hits = [c for c in scope if mentions(s, c, table)]
                    if hits:
                        owners, how = hits, "name_in_sentence"
                    elif section_owner and st in SECTIONED:
                        owners, how = [section_owner], "section_heading"
                    elif len(named) == 1 and st in SINGLE_TAG_OK:
                        owners, how = named, "source_tagged_single_candidate"
                if how in ("single_candidate_page", "source_tagged_single_candidate", "section_heading") and \
                        about_someone_else(s, owners[0]):
                    hits = [c for c in cmap if mentions(s, c, table)]
                    owners, how = (hits, "name_in_sentence") if hits else ([], None)
                if not owners:
                    if st == "party_platform":
                        claims.append(mk(None, party_id_of(party, cands), s, p, src, date, date_basis, "party_page"))
                    continue
                for o in owners[:3]:
                    claims.append(mk(o, None, s, p, src, date, date_basis, how))
    claims = shared_language(claims, cmap)
    claims = dedup(claims)
    with (OUT / "claims.jsonl").open("w") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    stats = defaultdict(int)
    for c in claims:
        stats[c["evidence_type"]] += 1
    return dict(claims=len(claims), canonical=sum(c["dedup_canonical"] for c in claims), by_type=dict(stats))


def st_party(cid: str, cmap: dict) -> str | None:
    pid = cmap[cid]["party_id"]
    return None if pid == "none" else pid


def party_id_of(party: str | None, cands: list[dict]) -> str | None:
    if not party:
        return None
    pn = norm(party)
    for c in cands:
        if norm(c["party"]) == pn or norm(c.get("party_label_on_ballot") or "-") == pn:
            return c["party_id"]
    for c in cands:
        if pn and (pn in norm(c["party"]) or norm(c["party"]) in pn):
            return c["party_id"]
    return None


SPEAK_VERBS = r"(said|says|told|wrote|writes|argued|argues|added|believes?|wants?|will|would|supports?|opposes?|proposes?|pledged|promised|promises|plans?|calls? for|is calling|vowed|vows|intends?|thinks?|stated|noted)"
_CANDS: dict = {}


AMBIG_FIRST = {"victoria", "job", "armor", "summit", "mark", "guy", "grant", "will", "rob", "kat", "jay", "wah", "ken"}
_FIRST: dict = {}


def about_someone_else(sent: str, owner: str) -> bool:
    """On a page attributed to `owner`, does the sentence name a different roster candidate (full name, or a
    distinctive first name) without naming the owner?"""
    c = _CANDS.get(owner)
    if not c:
        return False
    parts = c["display_name"].split()
    if re.search(rf"\b({re.escape(parts[0])}|{re.escape(parts[-1])})\b", sent):
        return False
    for first, ids in _FIRST.items():
        if owner in ids or first.lower() in AMBIG_FIRST:
            continue
        if re.search(rf"\b{re.escape(first)}\b", sent):
            return True
    return False


def speaks(sent: str, cid: str) -> bool:
    """Is the candidate the speaker/subject of the sentence (not merely mentioned)?"""
    c = _CANDS.get(cid)
    if not c:
        return True
    names = [re.escape(c["display_name"]), re.escape(c["display_name"].split()[-1])]
    title = r"(?:(?:Mayor|Councillor|Coun\.|Commissioner|Trustee|candidate|Dr\.)\s+)?"
    for n in names:
        if re.search(rf"\b{title}{n}(?:'s)?\s+(?:\w+\s+){{0,2}}{SPEAK_VERBS}\b", sent, re.I):
            return True
        if re.search(rf"\b(said|says|told|according to|wrote)\s+{title}{n}\b", sent, re.I):
            return True
        if re.search(rf"[”\"]\s*[–—-]\s*{n}", sent):
            return True
    return False


def mk(cid, pid, sent, p, src, date, date_basis, how, quoted_doc=None) -> dict:
    et, vs = classify(sent, src["source_type"], how)
    if cid and how == "name_in_sentence" and et == "individual_position" and not speaks(sent, cid):
        # mentioned, not speaking: a statement ABOUT the candidate, never their position
        et, vs = ("reported_action", vs) if src["source_type"] in ("news", "reference", "official_record") \
            else ("biography", vs)
    if cid is None:
        et, vs = "party_commitment", ("party_commitment" if how == "party_page" else
                                      "party_commitment_quoted_elsewhere")
    elif quoted_doc:
        vs = "aggregator_quote_of_candidate_source"
    if et == "party_commitment" and cid:
        # a platform passage naming a candidate is still a party commitment, never personal record
        pass
    tps = topics_of(sent)
    unc = {"single_candidate_page": "low", "party_page": "low", "section_heading": "medium",
           "source_tagged_single_candidate": "medium", "name_in_sentence": "medium",
           "aggregator_quote_of_party_document": "medium", "party_subject_sentence": "low"}[how]
    if src.get("fetch", {}).get("via") == "wayback":
        unc = {"low": "medium", "medium": "medium-high"}.get(unc, unc)
    if src["source_type"] == "commentary":
        unc = "high"
    cid_key = cid or f"party:{pid}"
    h = hashlib.sha1(f"{cid_key}|{p['passage_id']}|{sent}".encode()).hexdigest()[:12]
    return dict(claim_id="C" + h, candidate_id=cid, party_id=pid, topic=tps[0] if tps else "general",
                topics=tps, claim=sent, evidence_type=et, source_id=src["source_id"],
                passage_id=p["passage_id"], supporting_passage=p["text"][:1500], page=p.get("page"),
                date=date, date_basis=date_basis, verification_status=vs, uncertainty=unc,
                attribution=how, housing_forms=housing_forms(sent),
                management_signal=bool(MGMT.search(sent)), source_type=src["source_type"],
                source_duplicate_of=src.get("duplicate_of"), quoted_document=quoted_doc)


def shared_language(claims: list[dict], cmap: dict) -> list[dict]:
    """Identical campaign text attributed to two or more candidates of the same party is party
    language, not an individual position: retype it so it never enters individual vectors."""
    groups = defaultdict(set)
    for c in claims:
        if c["candidate_id"] and c["evidence_type"] == "individual_position" and c["source_type"] != "news":
            groups[(cmap[c["candidate_id"]]["party_id"], norm(c["claim"]))].add(norm(cmap[c["candidate_id"]]["display_name"]))
    for c in claims:
        if c["candidate_id"] and c["evidence_type"] == "individual_position" and c["source_type"] != "news":
            k = (cmap[c["candidate_id"]]["party_id"], norm(c["claim"]))
            me = norm(cmap[c["candidate_id"]]["display_name"])
            if k[0] != "none" and len(groups[k] - {me}) >= 1:  # other PEOPLE, not the same person's other ballot entry
                c["evidence_type"] = "shared_campaign_language"
                c["shared_with"] = sorted(groups[k] - {me})
    return claims


def dedup(claims: list[dict]) -> list[dict]:
    """Collapse repeated language for the same subject (candidate or party).

    The first occurrence (earliest-published, then non-wayback) is canonical;
    later copies keep their citations in `also_in_sources` and are flagged
    dedup_canonical=False so they never add weight to a representation."""
    groups = defaultdict(list)
    for c in claims:
        subj = c["candidate_id"] or f"party:{c['party_id']}"
        key = hashlib.sha1(f"{subj}|{norm(c['claim'])}".encode()).hexdigest()[:12]
        c["dedup_group"] = key
        groups[key].append(c)
    out = []
    for key, g in groups.items():
        g.sort(key=lambda c: (c["source_duplicate_of"] is not None, c["date"] or "9999"))
        for i, c in enumerate(g):
            c["dedup_canonical"] = i == 0
            c["also_in_sources"] = sorted({x["source_id"] for x in g if x is not c})
            out.append(c)
    return out


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
