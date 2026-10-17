"""Print a reproducible random sample of claims beside their source passage for manual checking.

usage: python scripts/spot_check.py N [seed]
Record the verdicts in data/validation/spot_check.md.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
claims = [json.loads(l) for l in open(ROOT / "data/processed/claims.jsonl")]
claims = [c for c in claims if c["dedup_canonical"] and c["evidence_type"] != "commentary"]
src = {s["source_id"]: s for s in json.loads((ROOT / "data/processed/sources.json").read_text())}
random.Random(seed).shuffle(claims)
for i, c in enumerate(claims[:n], 1):
    s = src[c["source_id"]]
    print(f"--- [{i}] {c['claim_id']}  {c['candidate_id'] or 'party:' + str(c['party_id'])}")
    print(f"type={c['evidence_type']} topic={c['topic']} status={c['verification_status']} unc={c['uncertainty']} attr={c['attribution']}")
    print(f"source: {s['publisher']} | {s['source_type']} | {s['url']}")
    print(f"CLAIM:   {c['claim']}")
    print(f"PASSAGE: {c['supporting_passage'][:600]}")
