.PHONY: install test pipeline

install:
	poetry install --with dev

test:
	poetry run pytest

# below are optional development utilities

pipeline:
	PYTHONWARNINGS=ignore poetry run python scripts/pipeline.py


