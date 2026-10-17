# Vancouver 2026 Candidate Explorer

An interactive website and Python pipeline for exploring the candidates in Vancouver's
2026-10-17 municipal election (mayor, council, Park Board, school trustees). It builds maps
from sourced evidence using text embeddings and unsupervised dimensionality reduction.

It does **not** endorse candidates, infer users' beliefs, or score candidate quality. Map axes
have no political meaning.

## Quick start

```bash
make setup            # uv venv + CPU torch + requirements.txt (Python 3.12)
make build            # full pipeline; network needed for uncached sources
make test             # validation suite
make serve            # http://127.0.0.1:8765
```

`make rebuild-analysis` re-runs claims → embeddings → analysis → exports from the cache, with
no network access.

## Pipeline (`python -m vanelect <step>`)

| step | module | output |
|---|---|---|
| ballot | `ballot.py`, `electionsbc.py` | `data/processed/candidates.json`, `ballot_meta.json` |
| votes | `votes.py` | `votes_summary.json`, `votes_candidates.csv` (City open data) |
| sources | `sources.py`, `fetch.py` | `sources.json`, `passages.jsonl`, `data/cache/`, `data/text/`, `data/access_log.jsonl` |
| claims | `claims.py` | `claims.jsonl` (atomic, verbatim, rule-classified) |
| embed | `represent.py`, `embed.py` | `chunks.jsonl`, `chunk_vectors.npy`, `repr_<view>.npz` |
| analyze | `analyze.py` | `data/site/analysis.json` (PCA, UMAP, neighbours, stability) |
| export | `export.py` | `data/site/*`, `data/exports/*`, methods report |

Run `python -m vanelect all` for everything, or `<step>+` to run from that step onward.
Settings live in `config/analysis.yaml`. Its hash is recorded in `meta.json`.

### Embedding model

The default is the local `sentence-transformers` model `BAAI/bge-small-en-v1.5` (384 dims). Its
exact revision is recorded in `data/processed/embedding_info.json`.

To use an API model, set `embedding.backend: openai-compatible` and `embedding.model` in the config.
Put `EMBEDDING_API_KEY` (and optionally `EMBEDDING_API_BASE`) in the **server** environment. Keys never
reach the browser. Vectors are cached per model, so switching models re-embeds.

### Source discovery

`data/seeds/*.json` lists the URLs found by research agents. They follow the brief in
`data/seeds/AGENT_BRIEF.md`. Add a URL there and re-run `make build`; each new URL is fetched once
and then cached.

## Provenance rules

- vancouver.ca, council.vancouver.ca and several party sites return Cloudflare bot challenges.
  These are **not** circumvented. The fallback is a public Internet Archive snapshot. Every refusal
  is logged.
- Ballot sources:
  - Mayor, Park Board and school lists come from archived copies of the City's official pages.
  - Council ballot numbers come from Daily Hive's ballot-order list, which matches the official pages
    on all 74 other entries.
  - All 130 names and offices are matched against Elections BC's official registered-candidate list.
- Each claim is a verbatim sentence with a passage id. Each source keeps its original bytes, URL,
  publisher, date, retrieval time and SHA-256 hash.
- Missing values stay distinct and are never zero-filled:
  - `not_found` and `party_platform_only` for topics;
  - `none_no_council_seat` for vote records;
  - the per-vote values Absent, Abstain, No Vote, Declared Conflict and Ineligible.

  None of these counts as a vote against.

See the methods report (`/methods` in the app, or `data/exports/methods_report.md`) for the full
method, coverage and limitations.

## Deployment on this box

Run the server, then serve it through Caddy:

```bash
setsid nohup .venv/bin/python -m vanelect serve 8765 > server.log 2>&1 &
```

The Caddy snippet goes in `~/sites/vanelection.caddy`.
