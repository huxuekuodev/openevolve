# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenEvolve is an open-source implementation of Google DeepMind's AlphaEvolve system - an evolutionary coding agent that uses LLMs to optimize code through iterative evolution. The framework can evolve code in multiple languages (Python, R, Rust, etc.) for tasks like scientific computing, optimization, and algorithm discovery.

## Essential Commands

### Development Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Or use Makefile
make install
```

### Running Tests
```bash
# Full suite; tests/integration/ skips itself when no optillm server is reachable
make test
python -m pytest

# Unit tests only (fast, no LLM required)
make test-unit
python -m pytest tests --ignore=tests/integration

# Coverage
make test-cov
python -m pytest tests --ignore=tests/integration --cov=openevolve --cov-report=term-missing

# Legacy unittest discovery (still supported)
make test-unittest
python -m unittest discover tests

# Integration tests against a local optillm server
make test-integration        # starts optillm itself
make test-integration-dev    # uses an already-running server on :8000
```

`OPENAI_API_KEY` must be set to any non-empty value for unit tests (no real API
calls are made): `export OPENAI_API_KEY=test`.

### Type Checking
```bash
make typecheck    # mypy; configured in pyproject.toml (files = ["openevolve"])
python -m mypy
```
mypy is clean and is a hard CI gate. `# type: ignore` is not used anywhere in
this codebase — fix the annotation or add the missing `None` guard instead.

### Code Formatting
```bash
make format         # isort + Black (rewrites files)
make format-check   # verify only, no writes

# Or directly
python -m black openevolve examples tests scripts
```

### PR Gate
```bash
make check    # typecheck + unit tests
```

### Running OpenEvolve
```bash
# Basic evolution run
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 1000

# Resume from checkpoint
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50
```

### Visualization
```bash
# View evolution tree
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

## High-Level Architecture

### Core Components

1. **Controller (`openevolve/controller.py`)**: Main orchestrator that manages the evolution process using ProcessPoolExecutor for parallel iteration execution.

2. **Database (`openevolve/database.py`)**: Implements MAP-Elites algorithm with island-based evolution:
   - Programs mapped to multi-dimensional feature grid
   - Multiple isolated populations (islands) evolve independently
   - Periodic migration between islands prevents convergence
   - Tracks absolute best program separately

3. **Evaluator (`openevolve/evaluator.py`)**: Cascade evaluation pattern:
   - Stage 1: Quick validation
   - Stage 2: Basic performance testing  
   - Stage 3: Comprehensive evaluation
   - Programs must pass thresholds at each stage

4. **LLM Integration (`openevolve/llm/`)**: Ensemble approach with multiple models, configurable weights, and async generation with retry logic.

5. **Iteration (`openevolve/process_parallel.py`)**: Worker process that samples from islands, generates mutations via LLM, evaluates programs, and returns results to the controller (which owns all database writes).

### Key Architectural Patterns

- **Island-Based Evolution**: Multiple populations evolve separately with periodic migration
- **MAP-Elites**: Maintains diversity by mapping programs to feature grid cells
- **Artifact System**: Side-channel for programs to return debugging data, stored as JSON or files
- **Process Worker Pattern**: Each iteration runs in fresh process with database snapshot
- **Double-Selection**: Programs for inspiration differ from those shown to LLM
- **Lazy Migration**: Islands migrate based on generation counts, not iterations

### Code Evolution Markers

Mark code sections to evolve using:
```python
# EVOLVE-BLOCK-START
# Code to evolve goes here
# EVOLVE-BLOCK-END
```

### Configuration

YAML-based configuration with hierarchical structure:
- LLM models and parameters
- Evolution strategies (diff-based vs full rewrites)
- Database and island settings
- Evaluation parameters

### Important Patterns

1. **Checkpoint/Resume**: Automatic saving of entire system state with seamless resume capability
2. **Parallel Evaluation**: Multiple programs evaluated concurrently via TaskPool
3. **Error Resilience**: Individual failures don't crash system - extensive retry logic and timeout protection
4. **Prompt Engineering**: Template-based system with context-aware building and evolution history

### Development Notes

- Python >=3.10 required
- Uses OpenAI-compatible APIs for LLM integration
- Tests run under pytest (which also collects the unittest-style `TestCase` classes); `make check` is the PR gate
- mypy must stay clean; `# type: ignore` is not used in this codebase
- Black (line-length 100) + isort for formatting; the repo has some pre-existing formatting drift, so `make format-check` is documented but not yet a CI gate
- Artifacts threshold: artifacts under `database.artifact_size_threshold` (default 32KB) are stored in the DB, larger ones are saved to disk
- Process workers load database snapshots for true parallelism; only the controller writes to the database