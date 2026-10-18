"""Remove browser runtime scripts from static HTML research snapshots.

The extraction pipeline reads server-rendered article text, not JavaScript.
Keep JSON-LD provenance, but avoid redistributing third-party runtime config,
API keys, trackers and hydration payloads in committed snapshots.
"""
import re
import sys
from pathlib import Path

SCRIPTS = re.compile(rb'<script\b[^>]*>.*?</script\s*>', re.I | re.S)
JSON_LD = re.compile(rb'\btype\s*=\s*[\"\x27]application/ld\+json[\"\x27]', re.I)


def sanitize(raw: bytes) -> bytes:
    def replacement(match):
        opening = match[0].split(b'>', 1)[0]
        return match[0] if JSON_LD.search(opening) else b''
    return SCRIPTS.sub(replacement, raw)


if __name__ == '__main__':
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        (Path(__file__).resolve().parents[1] / 'data/official_snapshots').glob('*.html'))
    for path in paths:
        raw = path.read_bytes()
        cleaned = sanitize(raw)
        if cleaned != raw:
            path.write_bytes(cleaned)
            print(f'Sanitized {path.name}')
