.PHONY: install test validate

install:
	poetry install --with dev

test:
	poetry run pytest

# below are optional development utilities

validate:
	poetry run python scripts/pipeline_validation.py


