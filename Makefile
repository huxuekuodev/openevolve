# Variables
PROJECT_DIR := $(shell pwd)
DOCKER_IMAGE := openevolve
# Prefer the .venv that `python -m venv .venv` / `uv sync` creates; fall back to
# the historical `env/` directory used by older checkouts and docs.
VENV_DIR ?= $(if $(wildcard $(PROJECT_DIR)/.venv/bin/python),$(PROJECT_DIR)/.venv,$(PROJECT_DIR)/env)
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# `pytest` is the primary test runner: it runs the whole unittest-style suite
# plus the plain-function tests, and adds per-test reporting, markers and
# coverage. Integration tests skip themselves when no optillm server is
# reachable; set OPENEVOLVE_REQUIRE_LLM_SERVER=1 to turn that skip into a failure.
PYTEST := $(PYTHON) -m pytest
PYTEST_UNIT := $(PYTEST) tests --ignore=tests/integration

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all              - Install dependencies and run unit tests"
	@echo "  venv             - Create a virtual environment"
	@echo "  install          - Install Python dependencies"
	@echo "  install-dev      - Install development dependencies including optillm"
	@echo "  lint             - Run Black code formatting"
	@echo "  format           - Format code with isort + Black"
	@echo "  format-check     - Check formatting without rewriting files (CI mode)"
	@echo "  typecheck        - Run mypy over the openevolve package"
	@echo "  check            - typecheck + unit tests"
	@echo "  test             - Run the full suite (integration tests skip without an LLM server)"
	@echo "  test-unit        - Run unit tests only (fast, no LLM required)"
	@echo "  test-unittest    - Run unit tests via unittest discovery (legacy runner)"
	@echo "  test-cov         - Run unit tests with a coverage report"
	@echo "  test-integration - Run integration tests with local LLM"
	@echo "  test-all         - Run both unit and integration tests"
	@echo "  docker-build     - Build the Docker image"
	@echo "  docker-run       - Run the Docker container with the example"
	@echo "  visualizer       - Run the visualization script"

.PHONY: all
all: install test

# Create and activate the virtual environment
.PHONY: venv
venv:
	python3 -m venv $(VENV_DIR)

# Install Python dependencies in the virtual environment
.PHONY: install
install: venv
	$(PIP) install -e .

# Install development dependencies including optillm for integration tests
.PHONY: install-dev
install-dev: venv
	$(PIP) install -e ".[dev]"
	$(PIP) install optillm math-verify

# Run Black code formatting
.PHONY: lint
lint: venv
	$(PYTHON) -m black openevolve examples tests scripts

# Format code (isort then Black, matching .pre-commit-config.yaml)
.PHONY: format
format: venv
	$(PYTHON) -m isort --profile black openevolve examples tests scripts
	$(PYTHON) -m black openevolve examples tests scripts

# Verify formatting in CI without rewriting files
.PHONY: format-check
format-check: venv
	$(PYTHON) -m isort --profile black --check-only --diff openevolve tests scripts
	$(PYTHON) -m black --check --diff openevolve tests scripts

# Static type checking
.PHONY: typecheck
typecheck: venv
	$(PYTHON) -m mypy

# Everything a pull request must pass. `format-check` is intentionally not part
# of this target: the repo has pre-existing black/isort drift (see the note in
# .github/workflows/python-test.yml), so gating on it would fail immediately.
# Once `make format` has been run repo-wide, add `format-check` here.
.PHONY: check
check: typecheck test-unit

# Run the full suite; integration tests skip when no optillm server is available
.PHONY: test
test: venv
	$(PYTEST)

# Unit tests only (fast, no LLM required)
.PHONY: test-unit
test-unit: venv
	$(PYTEST_UNIT)

# Legacy runner: `python -m unittest` cannot run the plain-function tests and
# does not understand the pytest fixtures, but it is kept working for
# downstream tooling that calls it directly.
.PHONY: test-unittest
test-unittest: venv
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

# Unit tests with coverage (XML report for CI annotations)
.PHONY: test-cov
test-cov: venv
	$(PYTEST_UNIT) --cov=openevolve --cov-report=term-missing --cov-report=xml

# Run integration tests with local LLM (requires optillm)
.PHONY: test-integration
test-integration: install-dev
	@echo "Starting optillm server for integration tests..."
	@OPTILLM_API_KEY=optillm $(VENV_DIR)/bin/optillm --model google/gemma-3-270m-it --port 8000 &
	@OPTILLM_PID=$$! && \
	echo $$OPTILLM_PID > /tmp/optillm.pid && \
	echo "Waiting for optillm server to start..." && \
	sleep 10 && \
	echo "Running integration tests..." && \
	OPENAI_API_KEY=optillm OPTILLM_API_KEY=optillm OPENEVOLVE_REQUIRE_LLM_SERVER=1 \
		$(PYTEST) tests/integration -v --tb=short; \
	TEST_EXIT_CODE=$$?; \
	echo "Stopping optillm server..."; \
	kill $$OPTILLM_PID 2>/dev/null || true; \
	pkill -f "optillm.*8000" 2>/dev/null || true; \
	rm -f /tmp/optillm.pid; \
	exit $$TEST_EXIT_CODE

# Run integration tests with existing optillm server (for development)
.PHONY: test-integration-dev
test-integration-dev: venv
	@echo "Using existing optillm server at localhost:8000"
	@curl -s http://localhost:8000/health > /dev/null || (echo "Error: optillm server not running at localhost:8000" && exit 1)
	OPENAI_API_KEY=optillm OPENEVOLVE_REQUIRE_LLM_SERVER=1 $(PYTEST) tests/integration -v

# Run all tests (unit first, then integration)
.PHONY: test-all
test-all: test test-integration

# Build the Docker image
.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# Run the Docker container with the example
.PHONY: docker-run
docker-run:
	docker run --rm -v $(PROJECT_DIR):/app --network="host" $(DOCKER_IMAGE) examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config.yaml --iterations 1000

# Run the visualization script
.PHONY: visualizer
visualizer:
	$(PYTHON) scripts/visualizer.py --path examples/
