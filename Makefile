.PHONY: install test

install:
	poetry install --with dev

test:
	poetry run pytest


