NAME := rnd-convergence
PACKAGE_NAME := rnd_convergence

.PHONY: help install check format test

help:
	@echo "Makefile ${NAME}"
	@echo "* install      install all requirements and pre-commit hooks"
	@echo "* check        check the source code for formatting and lint issues"
	@echo "* format       format the code with ruff"
	@echo "* test         run the tests"

PIP ?= uv pip
PYTEST ?= uv run pytest
RUFF ?= uv run ruff

install:
	$(PIP) install swig
	$(PIP) install -e ".[dev]"

check:
	$(RUFF) format --check $(PACKAGE_NAME) tests scripts
	$(RUFF) check $(PACKAGE_NAME) tests scripts

format:
	$(RUFF) format $(PACKAGE_NAME) tests scripts
	$(RUFF) check --fix $(PACKAGE_NAME) tests scripts

test:
	$(PYTEST) tests
