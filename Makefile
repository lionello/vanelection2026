PY := .venv/bin/python
PORT ?= 8765

.PHONY: setup build rebuild-analysis serve test spot-check clean-derived

setup:            ## create venv and install dependencies (CPU torch)
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python $(PY) -r requirements.txt

build:            ## full pipeline: ballot -> votes -> sources -> claims -> embed -> analyze -> export
	$(PY) -m vanelect all

rebuild-analysis: ## re-run claims onward from cached sources (no network)
	$(PY) -m vanelect claims+

serve:
	$(PY) -m vanelect serve $(PORT)

test:
	$(PY) -m pytest -q tests

spot-check:       ## print a random sample of claims next to their source passages
	$(PY) scripts/spot_check.py 25

clean-derived:    ## remove derived data (keeps fetch cache and official inputs)
	rm -rf data/processed data/site data/exports data/text
