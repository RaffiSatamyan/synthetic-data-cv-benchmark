.PHONY: install test lint pilot

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

pilot:
	python -m synthbench.experiments --experiment experiments/pilot.yaml --output experiments/pilot_runs.csv
