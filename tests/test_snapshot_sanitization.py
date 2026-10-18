import importlib.util
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sanitize_snapshots', ROOT / 'scripts/sanitize_snapshots.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_runtime_scripts_removed_without_changing_static_text():
    raw = b'''<html><head><script type="application/ld+json">{"author":"Example"}</script>
<script id="__NEXT_DATA__" type="application/json">{"runtimeConfig":{"ALGOLIA_SEARCH_API_KEY":"dummy-not-a-real-key"}}</script>
<script src="tracker.js"></script></head><body><h1>Candidate list</h1><p>100: Example Name</p></body></html>'''
    clean = module.sanitize(raw)
    assert b'dummy-not-a-real-key' not in clean
    assert b'tracker.js' not in clean
    assert b'"author":"Example"' in clean
    assert BeautifulSoup(raw, 'lxml').get_text().split() == BeautifulSoup(clean, 'lxml').get_text().split()
    assert module.sanitize(clean) == clean


def test_dailyhive_snapshot_contains_no_browser_runtime():
    raw = (ROOT / 'data/official_snapshots/dailyhive-ballot-order-2026-09-22.html').read_bytes()
    assert b'ALGOLIA_SEARCH_API_KEY' not in raw
    assert b'__NEXT_DATA__' not in raw
