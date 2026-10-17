"""python -m vanelect <step>   steps: ballot votes sources claims embed analyze export all serve"""
import json
import sys
import time

from . import ballot, votes, sources, board_votes, claims, represent, analyze, export

STEPS = [("ballot", ballot.build), ("votes", votes.build), ("sources", sources.build), ("board_votes", board_votes.build), ("claims", claims.build),
         ("embed", represent.build), ("analyze", analyze.build), ("export", export.build)]


def main(argv):
    what = argv[1] if len(argv) > 1 else "all"
    if what == "serve":
        import uvicorn
        port = int(argv[2]) if len(argv) > 2 else 8765
        uvicorn.run("vanelect.server:app", host="127.0.0.1", port=port)
        return
    todo = STEPS if what == "all" else [s for s in STEPS if s[0] == what]
    if what.endswith("+"):
        names = [s[0] for s in STEPS]
        todo = STEPS[names.index(what[:-1]):]
    if not todo:
        sys.exit(f"unknown step {what}; choose from {[s[0] for s in STEPS]} or all / <step>+ / serve")
    for name, fn in todo:
        t = time.time()
        print(f"== {name}", flush=True)
        r = fn()
        out = json.dumps(r, ensure_ascii=False, default=str)
        print(out[:1500], flush=True)
        print(f"   ({time.time() - t:.1f}s)", flush=True)


main(sys.argv)
