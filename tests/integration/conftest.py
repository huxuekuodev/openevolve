"""
Pytest fixtures for integration tests with optillm server
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Import our test utilities
import sys

sys.path.append(str(Path(__file__).parent.parent))
from test_utils import (
    start_test_server,
    stop_test_server,
    is_server_running,
    get_integration_config,
    get_evolution_test_program,
    get_evolution_test_evaluator,
)

# When set to a truthy value, an unavailable optillm server is a hard failure
# instead of a skip. CI sets this so a broken server can never silently turn the
# whole integration suite green.
REQUIRE_SERVER_ENV = "OPENEVOLVE_REQUIRE_LLM_SERVER"

_SKIP_REASON = (
    "optillm server unavailable (not running on localhost:8000 and could not be "
    f"started). Install optillm and start it, or set {REQUIRE_SERVER_ENV}=1 to make "
    "this a hard failure."
)


def pytest_collection_modifyitems(config, items):
    """Tag everything collected below tests/integration/ with the `integration` marker."""
    marker = pytest.mark.integration
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(marker)


def _require_or_skip(reason: str) -> None:
    """Skip when the LLM server is only unavailable, fail when it is required."""
    if os.environ.get(REQUIRE_SERVER_ENV, "").strip().lower() in {"1", "true", "yes"}:
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def _server_unavailable(reason: str) -> None:
    """No optillm server and none can be started."""
    _require_or_skip(f"{_SKIP_REASON} ({reason})")


@pytest.fixture(scope="session")
def optillm_server():
    """Start optillm server for the test session"""
    # Check if server is already running (for development)
    if is_server_running(8000):
        print("Using existing optillm server at localhost:8000")
        yield {"proc": None, "port": 8000}  # Server already running, don't manage it
        return

    # A missing `optillm` executable must skip, not explode inside subprocess.
    if shutil.which("optillm") is None:
        _server_unavailable("the `optillm` executable is not on PATH")
        return

    print("Starting optillm server for integration tests...")
    proc = None
    port = None
    try:
        proc, port = start_test_server()
        print(f"optillm server started successfully on port {port}")
        yield {"proc": proc, "port": port}
    except Exception as e:
        # start_test_server already cleaned up its process on failure.
        _server_unavailable(str(e))
        return
    finally:
        if proc:
            print("Stopping optillm server...")
            stop_test_server(proc)
            print("optillm server stopped")


@pytest.fixture
def evolution_config(optillm_server):
    """Get config for evolution tests"""
    port = optillm_server["port"]
    return get_integration_config(port)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_program_file(temp_workspace):
    """Create a test program file"""
    program_file = temp_workspace / "test_program.py"
    program_file.write_text(get_evolution_test_program())
    return program_file


@pytest.fixture
def test_evaluator_file(temp_workspace):
    """Create a test evaluator file"""
    evaluator_file = temp_workspace / "evaluator.py"
    evaluator_file.write_text(get_evolution_test_evaluator())
    return evaluator_file


@pytest.fixture
def evolution_output_dir(temp_workspace):
    """Create output directory for evolution tests"""
    output_dir = temp_workspace / "output"
    output_dir.mkdir()
    return output_dir
