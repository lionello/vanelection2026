# Source-discovery brief (for research subagents)

Project: an evidence explorer for the Vancouver municipal election of 2026-10-17. The pipeline
(written by the lead agent) fetches, caches and cites every URL you record. Your job is ONLY to
discover and record source URLs plus a few verified facts. Do not write code in vanelect/.

Roster: data/seeds/ROSTER.txt (candidate_id, ballot #, office, name, grouped by ballot party).
Use those candidate_ids exactly.

## Rules
- Respect access restrictions. Many sites (vancouver.ca, council.vancouver.ca, abcvancouver.ca,
  vangreens.ca, vancouverliberals.com) return a Cloudflare 403 challenge to bots. Do NOT try to get
  around it (no header spoofing tricks, no headless browsers, no proxies). For those, look for an
  Internet Archive copy: `curl -s "https://web.archive.org/cdx/search/cdx?url=URL&output=json&fl=timestamp,statuscode&filter=statuscode:200"`
  and record the original URL with "access": "blocked_use_wayback" and "wayback_ts" set.
  Be gentle with archive.org (no more than ~1 request every 2 seconds; it rate-limits).
- Record what you could actually open. If a page you expected cannot be found at all, record it
  under "gaps" with what you tried. Never invent URLs, dates, roles or quotes.
- Prefer: candidate's own bio page on the party site; candidate's own campaign site; party platform
  pages (one entry per distinct platform page/PDF, NOT per candidate); official records
  (council/park/school board minutes, City pages for elected officials, employer bios, LinkedIn is
  NOT fetchable - skip it); credible news (Vancouver Sun, CBC, CTV, Global, Georgia Straight,
  Daily Hive, BIV/Vancouver Is Awesome, The Tyee, CityNews, Glacier/Vancouver Courier,
  Ming Pao etc.) about concrete actions/outcomes; questionnaires and advocacy-group report cards.
- Party platforms: record each platform page/PDF once with "candidates": [] and "party" set.
- Commentary (e.g. allenpike.com guide) is "commentary".

## Output
Write ONE JSON file at the path you are given: an object
{
  "sources": [
    {"url": "...", "title": "...", "publisher": "...", "published": "YYYY-MM-DD or null",
     "source_type": "candidate_bio|campaign_site|party_platform|party_candidate_list|official_record|employer_bio|news|questionnaire|advocacy_assessment|commentary|reference",
     "candidates": ["candidate_id", ...],   // whose evidence this is; [] for party-wide
     "party": "party name or null",
     "topics": ["housing","homelessness","drug_policy","policing","active_transport","transit","taxes","public_services","parks","education","governance","displacement","coops_public_housing", ...],
     "perspective": "for questionnaires/assessments/commentary: who publishes it and their stated stance; else null",
     "access": "direct|blocked_use_wayback",
     "wayback_ts": "only if blocked_use_wayback",
     "notes": "one line: what is in it"}
  ],
  "affiliations": [
    {"candidate_id": "...", "organization": "...", "role": "e.g. member/councillor/candidate",
     "start": "YYYY or YYYY-MM or null", "end": "... or null (current)", "source_url": "...",
     "kind": "party|elected_office|board|employer"}
  ],
  "gaps": [{"candidate_id": "...", "what": "...", "tried": "..."}]
}
Aim for: every candidate in your set has at least one bio-type source if one exists anywhere; every
party platform page; elected-office history for incumbents/former officials. Finish within ~60-90
minutes of work; breadth across all your candidates matters more than depth on a few. Validate the
JSON parses (python3 -m json.tool FILE) before finishing. Reply with a 5-line summary.
