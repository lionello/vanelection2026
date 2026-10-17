# Manual spot-checks of extracted claims

Method: `python scripts/spot_check.py 25 <seed>` prints a reproducible random sample of canonical,
non-commentary claims beside their source passage. Each claim was read against its passage and its
source page, and judged on attribution (right subject), evidence type, and verbatim accuracy.
Verbatim accuracy for *all* claims is enforced by `tests/test_pipeline.py::test_claims_are_verbatim_and_resolve`.

## Round 1 (seed 7, 238 sources, 2026-10-17 ~12:30 UTC)
Error rate about 10/25. Main defects, all fixed:
- VoteMate profiles quote party platforms ("— Fight for Vancouver", "— Vancouver Green Party Platform, p.33")
  under each candidate, and those quotes were counted as individual positions. **Fix:** attribution lines are parsed;
  quotes of party documents become `party_commitment`, and unattributed quotes on aggregator pages are skipped.
- Advocacy report-card sentences about a candidate were typed as that candidate's position.
  **Fix:** new type `third_party_assessment`, shown with publisher perspective and never embedded.
- Footers / ads ("[Sponsored]", land acknowledgement), citation-title lines.

## Round 2 (seed 2026)
Error rate about 10/25. Defects, all fixed:
- News articles tagged with one candidate had every sentence attributed to that candidate (e.g. a parent's
  complaint attributed to a former superintendent). **Fix:** news, reference, official and employer pages need the
  candidate named in the sentence.
- Questionnaire *questions* were extracted as answers. **Fix:** sentences ending in "?" are dropped.
- "endorsed Ken Sim for mayor in 2022" was typed as elected experience. **Fix:** tighter elected-office rule
  (needs a served/elected verb).
- PDF line-break hyphenation ("Com- munity"). **Fix:** de-hyphenation at extraction.
- Verbatim text repeated across a party's candidates. **Fix:** `shared_campaign_language`, excluded from
  individual vectors, plus a cross-candidate near-duplicate check (cosine >= 0.95).

## Round 3 (seed 99, all 484 sources, final rules)
19/25 fully correct; 0 wrong-candidate attributions after the fix below.
- 1 attribution error, fixed: the same person on two ballots (Naomi Chocyk, council and school) had her own text
  flagged as "shared with another candidate". Shared language now requires a different *person*.
- 2 trivial non-claims ("John is running for City Councillor"), now filtered.
- 2 biographical sentences typed as `individual_position` / `reported_action` (VSB trustee bio page, TEAM park
  candidate bio). Not fixed: a known limitation of rule-based typing. Uncertainty is "medium", and the citation
  shows the context.
- 1 claim keeps an HTML artifact ("</a>") from VoteMate's own text. Not fixed: it is verbatim.

## Vote parsing
- Council (open data): 15 tests confirm that value sets are preserved, counts sum, pair overlap and agreement are
  recomputed independently from the per-vote file, and no pair spans different councils.
- Park Board minutes: the first parser attached a later motion's member list to an earlier unanimous motion
  (a "CARRIED UNANIMOUSLY" with three opposed). **Fix:** the lookahead stops at the next MOVED/result line, plus a
  regression test (no unanimous result with named opposition). A 4-3 "LOST" was checked against the minutes and is
  correct: it needed a 2/3 majority, a qualifier now kept in the result text.

## Duplicate handling
`test_duplicate_document_does_not_change_representation` re-adds every policy chunk of the most prolific
candidate as a new source (a simulated syndicated copy). Every topic vector keeps cosine > 0.999 with the
original, and all injected chunks are flagged `near_duplicate_of`.
