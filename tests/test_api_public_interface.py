"""Public-interface tests for ``openevolve/api.py``.

``tests/test_api.py`` covers the private preparation helpers and stubs out
``_run_evolution_async`` wholesale. This file deliberately goes further:

* it drives the *real* ``run_evolution`` / ``_run_evolution_async`` code paths,
  including config resolution, LLM-model validation, cascade auto-detection,
  result shaping and temp-file cleanup;
* it stops the run at the only genuinely external boundary -- the controller
  (``openevolve.controller.OpenEvolve``) is replaced by ``FakeController``, so
  no LLM client is ever constructed and no network call is possible;
* it covers the error/validation paths of ``evolve_function``, ``evolve_code``,
  ``evolve_algorithm`` and ``run_evolution``.
"""

import importlib.util
import inspect
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

import openevolve.api as api_module
from openevolve.api import (
    EvolutionResult,
    _extract_lambda_source,
    _prepare_evaluator,
    _prepare_program,
    evolve_algorithm,
    evolve_code,
    evolve_function,
    run_evolution,
)
from openevolve.config import Config, LLMModelConfig
from openevolve.database import Program

# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------


class FakeController:
    """Stand-in for ``openevolve.controller.OpenEvolve``.

    Records how the API configured it and returns a canned program, so the API
    logic is exercised end to end without any LLM/network involvement.
    """

    instances: List["FakeController"] = []
    next_result: Any = None
    run_raises: Any = None

    def __init__(self, **kwargs: Any):
        self.init_kwargs = kwargs
        self.run_kwargs: Dict[str, Any] = {}
        self.result = type(self).next_result
        FakeController.instances.append(self)

    async def run(self, iterations=None, target_score=None, checkpoint_path=None):
        self.run_kwargs = {
            "iterations": iterations,
            "target_score": target_score,
            "checkpoint_path": checkpoint_path,
        }
        if type(self).run_raises is not None:
            raise type(self).run_raises
        return self.result


@pytest.fixture
def fake_controller(monkeypatch: pytest.MonkeyPatch):
    FakeController.instances = []
    FakeController.next_result = None
    FakeController.run_raises = None
    monkeypatch.setattr(api_module, "OpenEvolve", FakeController)
    return FakeController


@pytest.fixture(autouse=True)
def run_in_tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the default relative output dir out of the repository."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def cleanup_generated_temp_files(fake_controller):
    """Guarantee that no generated temp file outlives its test.

    ``cleanup=False`` runs deliberately leave the auto-generated program and
    evaluator modules in the system temp directory; this removes them.
    """
    yield
    for instance in FakeController.instances:
        _cleanup_temp_files(instance)


def controller() -> FakeController:
    assert len(FakeController.instances) == 1, "expected exactly one controller"
    return FakeController.instances[0]


def make_config(cascade_evaluation: bool = True) -> Config:
    """A config with one (never-contacted) LLM model so validation passes."""
    config = Config()
    config.llm.models = [LLMModelConfig(name="gpt-4", api_key="test-key")]
    config.evaluator.cascade_evaluation = cascade_evaluation
    return config


def make_program(code: str = "def f():\n    return 1\n", metrics: Any = None) -> Program:
    if metrics is None:
        metrics = {"combined_score": 0.5}
    return Program(id="prog-1", code=code, metrics=metrics)


EVALUATOR_CODE = """
def evaluate(program_path):
    return {"combined_score": 0.9, "score": 0.9}
"""


def import_file(path: str) -> Any:
    """Import a generated python file as a fresh module."""
    spec = importlib.util.spec_from_file_location(f"generated_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exec_evaluator(code: str) -> Any:
    """Exec generated evaluator *source* and return its ``evaluate`` callable."""
    namespace: Dict[str, Any] = {}
    exec(compile(code, "<generated evaluator>", "exec"), namespace)
    return namespace["evaluate"]


def write_program(path: Path, source: str) -> str:
    path.write_text(source)
    return str(path)


# ---------------------------------------------------------------------------
# evolve_function / evolve_algorithm building blocks: module-level definitions
# (inspect.getsource only works for objects that live in a real file)
# ---------------------------------------------------------------------------


def bubble_sort(arr):
    """Deliberately simple initial implementation."""
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def with_markers(value):
    # EVOLVE-BLOCK-START
    return value + 1
    # EVOLVE-BLOCK-END


class SumAlgorithm:
    def process(self, data):
        return sum(data)


class MarkedAlgorithm:
    # EVOLVE-BLOCK-START
    def process(self, data):
        return sum(data)

    # EVOLVE-BLOCK-END


def benchmark_sum(instance):
    result = instance.process([1, 2, 3])
    return {"score": 1.0 if result == 6 else 0.0, "runtime": 0.5}


def benchmark_scalar(instance):
    return 0.25


def benchmark_with_combined(instance):
    return {"combined_score": 0.75}


# ---------------------------------------------------------------------------
# EvolutionResult
# ---------------------------------------------------------------------------


def test_evolution_result_field_order_and_values():
    program = make_program()
    metrics = {"combined_score": 0.75, "runtime": 1.5}

    result = EvolutionResult(program, 0.75, "code", metrics, "/tmp/out")

    assert [f.name for f in fields(EvolutionResult)] == [
        "best_program",
        "best_score",
        "best_code",
        "metrics",
        "output_dir",
    ]
    assert result.best_program is program
    assert result.best_score == 0.75
    assert result.best_code == "code"
    assert result.metrics is metrics
    assert result.output_dir == "/tmp/out"


def test_evolution_result_is_a_value_object():
    first = EvolutionResult(None, 0.5, "code", {"a": 1}, None)
    same = EvolutionResult(None, 0.5, "code", {"a": 1}, None)
    different = EvolutionResult(None, 0.6, "code", {"a": 1}, None)

    assert first == same
    assert first != different


def test_evolution_result_is_unhashable():
    """``eq=True`` without ``frozen=True`` clears ``__hash__`` on the dataclass."""
    result = EvolutionResult(None, 0.5, "code", {"a": 1}, None)

    with pytest.raises(TypeError, match="unhashable"):
        hash(result)


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.85, "EvolutionResult(best_score=0.8500)"),
        (1.0, "EvolutionResult(best_score=1.0000)"),
        (0, "EvolutionResult(best_score=0.0000)"),
        (0.123456, "EvolutionResult(best_score=0.1235)"),
        (-0.5, "EvolutionResult(best_score=-0.5000)"),
    ],
)
def test_evolution_result_repr_formats_the_score_to_four_decimals(score, expected):
    result = EvolutionResult(None, score, "", {}, None)

    assert repr(result) == expected
    assert str(result) == expected
    assert f"{result}" == expected


def test_evolution_result_repr_requires_a_number():
    """``__repr__`` formats with ``:.4f``, so a non-number blows up."""
    result = EvolutionResult(None, "not-a-score", "", {}, None)

    with pytest.raises((TypeError, ValueError)):
        repr(result)


# ---------------------------------------------------------------------------
# run_evolution: result shaping
# ---------------------------------------------------------------------------


def test_run_evolution_returns_the_best_program(fake_controller, tmp_path):
    program = make_program(code="def f():\n    return 42\n", metrics={"combined_score": 0.77})
    fake_controller.next_result = program

    result = run_evolution("def f():\n    return 1\n", EVALUATOR_CODE, config=make_config())

    assert isinstance(result, EvolutionResult)
    assert not inspect.isawaitable(result)  # run_evolution is the sync wrapper
    assert result.best_program is program
    assert result.best_code == program.code
    assert result.best_score == 0.77
    assert result.metrics == {"combined_score": 0.77}
    # cleanup=True (default) -> temp dirs are reported as None and removed
    assert result.output_dir is None
    prepared_program = Path(controller().init_kwargs["initial_program_path"])
    assert not prepared_program.exists()


def test_run_evolution_averages_numeric_metrics_without_combined_score(fake_controller):
    fake_controller.next_result = make_program(metrics={"score": 0.4, "runtime": 0.6})

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    assert result.best_score == pytest.approx(0.5)


def test_run_evolution_score_is_zero_when_metrics_are_not_numeric(fake_controller):
    fake_controller.next_result = make_program(metrics={"note": "no numbers here"})

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    assert result.best_score == 0.0


def test_run_evolution_does_not_count_a_boolean_metric_as_a_score(fake_controller):
    """A boolean metric flag must not contribute to ``best_score``.

    ``openevolve/evaluator.py`` reports failures as ``{"error": 0.0, "timeout": True}``.
    ``bool`` is a subclass of ``int``, so an ad-hoc ``isinstance(value, (int, float))``
    average scored a timed-out program 0.5 instead of 0.0. ``run_evolution`` now uses
    ``metrics_utils.safe_numeric_average``, which excludes booleans (see
    ``tests/test_boolean_metrics.py`` for the same rule elsewhere in the codebase).
    """
    fake_controller.next_result = make_program(metrics={"error": 0.0, "timeout": True})

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    assert result.best_score == 0.0


def test_run_evolution_averages_genuine_numeric_metrics(fake_controller):
    """Non-boolean numeric metrics are still averaged when combined_score is absent."""
    fake_controller.next_result = make_program(metrics={"accuracy": 0.8, "speed": 0.4})

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    assert result.best_score == pytest.approx(0.6)


def test_run_evolution_without_a_best_program_returns_an_empty_result(fake_controller):
    fake_controller.next_result = None

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    assert result.best_program is None
    assert result.best_score == 0.0
    assert result.best_code == ""
    assert result.metrics == {}
    assert result.output_dir is None


def test_run_evolution_with_empty_metrics_on_the_best_program(fake_controller):
    fake_controller.next_result = make_program(metrics={})

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    assert result.metrics == {}
    assert result.best_score == 0.0
    assert result.best_code == "def f():\n    return 1\n"


def test_run_evolution_forwards_run_arguments_to_the_controller(fake_controller):
    fake_controller.next_result = make_program()

    run_evolution(
        "def f(): pass",
        EVALUATOR_CODE,
        config=make_config(),
        iterations=7,
        target_score=0.99,
        checkpoint_path="/tmp/checkpoint",
    )

    assert controller().run_kwargs == {
        "iterations": 7,
        "target_score": 0.99,
        "checkpoint_path": "/tmp/checkpoint",
    }


def test_run_evolution_forwards_island_selector_and_population_strategy(fake_controller):
    fake_controller.next_result = make_program()
    selector = object()
    strategy = object()

    run_evolution(
        "def f(): pass",
        EVALUATOR_CODE,
        config=make_config(),
        island_selector=selector,
        population_strategy=strategy,
    )

    assert controller().init_kwargs["island_selector"] is selector
    assert controller().init_kwargs["population_strategy"] is strategy


# ---------------------------------------------------------------------------
# run_evolution: input handling
# ---------------------------------------------------------------------------


def test_run_evolution_uses_an_existing_program_file_directly(fake_controller, tmp_path):
    fake_controller.next_result = make_program()
    program_file = write_program(tmp_path / "program.py", "def f():\n    return 7\n")

    run_evolution(program_file, EVALUATOR_CODE, config=make_config())

    assert controller().init_kwargs["initial_program_path"] == program_file
    assert Path(program_file).exists()  # user files are never deleted


def test_run_evolution_accepts_path_objects(fake_controller, tmp_path):
    fake_controller.next_result = make_program()
    program_file = tmp_path / "program.py"
    program_file.write_text("def f():\n    return 7\n")
    eval_file = tmp_path / "evaluator.py"
    eval_file.write_text(EVALUATOR_CODE)

    run_evolution(program_file, eval_file, config=make_config())

    assert controller().init_kwargs["initial_program_path"] == str(program_file)
    assert controller().init_kwargs["evaluation_file"] == str(eval_file)


def test_run_evolution_wraps_a_code_string_in_evolve_markers(fake_controller):
    fake_controller.next_result = make_program()

    run_evolution("def f():\n    return 1\n", EVALUATOR_CODE, config=make_config(), cleanup=False)

    content = Path(controller().init_kwargs["initial_program_path"]).read_text()
    assert content.count("EVOLVE-BLOCK-START") == 1
    assert content.count("EVOLVE-BLOCK-END") == 1
    assert "def f():\n    return 1\n" in content


def test_run_evolution_keeps_existing_evolve_markers(fake_controller):
    fake_controller.next_result = make_program()
    code = "# EVOLVE-BLOCK-START\ndef f():\n    return 1\n# EVOLVE-BLOCK-END"

    run_evolution(code, EVALUATOR_CODE, config=make_config(), cleanup=False)

    content = Path(controller().init_kwargs["initial_program_path"]).read_text()
    assert content.count("EVOLVE-BLOCK-START") == 1
    assert content.count("EVOLVE-BLOCK-END") == 1


def test_run_evolution_joins_a_list_of_program_lines(fake_controller):
    fake_controller.next_result = make_program()

    run_evolution(["def f():", "    return 1"], EVALUATOR_CODE, config=make_config(), cleanup=False)

    content = Path(controller().init_kwargs["initial_program_path"]).read_text()
    assert "def f():\n    return 1" in content


def test_run_evolution_treats_a_missing_program_path_as_literal_code(fake_controller, tmp_path):
    """Real behaviour: a non-existent path is not an error, it becomes the source."""
    fake_controller.next_result = make_program()
    missing = str(tmp_path / "does_not_exist.py")

    run_evolution(missing, EVALUATOR_CODE, config=make_config(), cleanup=False)

    content = Path(controller().init_kwargs["initial_program_path"]).read_text()
    assert missing in content


def test_run_evolution_uses_an_existing_evaluator_file_directly(fake_controller, tmp_path):
    fake_controller.next_result = make_program()
    eval_file = write_program(tmp_path / "evaluator.py", EVALUATOR_CODE)

    run_evolution("def f(): pass", eval_file, config=make_config())

    assert controller().init_kwargs["evaluation_file"] == eval_file
    assert Path(eval_file).exists()


def test_run_evolution_writes_evaluator_source_unchanged(fake_controller):
    fake_controller.next_result = make_program()

    run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config(), cleanup=False)

    assert Path(controller().init_kwargs["evaluation_file"]).read_text() == EVALUATOR_CODE


def test_run_evolution_serializes_a_lambda_evaluator(fake_controller):
    fake_controller.next_result = make_program()
    evaluator = lambda program_path: {"combined_score": 0.1}  # noqa: E731

    run_evolution("def f(): pass", evaluator, config=make_config(), cleanup=False)

    generated = Path(controller().init_kwargs["evaluation_file"]).read_text()
    assert '_user_evaluator = lambda program_path: {"combined_score": 0.1}' in generated
    assert "return _user_evaluator(program_path)" in generated


def test_run_evolution_falls_back_to_module_globals_for_unsourceable_callables(
    fake_controller,
):
    """A callable with no retrievable source is registered on the api module."""
    fake_controller.next_result = make_program()
    before = set(vars(api_module))

    run_evolution("def f(): pass", CallableEvaluator(), config=make_config(), cleanup=False)

    generated = Path(controller().init_kwargs["evaluation_file"]).read_text()
    assert "import openevolve.api as api_module" in generated
    added = set(vars(api_module)) - before
    try:
        assert len(added) == 1
        attribute = added.pop()
        assert attribute.startswith("_openevolve_evaluator_")
        assert attribute in generated
        # The generated module really calls back into the registered callable.
        module = import_file(controller().init_kwargs["evaluation_file"])
        assert module.evaluate("/tmp/whatever.py") == {"combined_score": 0.125}
    finally:
        for attribute in set(vars(api_module)) - before:
            delattr(api_module, attribute)


# ---------------------------------------------------------------------------
# run_evolution: output directory and cleanup
# ---------------------------------------------------------------------------


def test_run_evolution_creates_a_missing_output_dir(fake_controller, tmp_path):
    fake_controller.next_result = make_program()
    output_dir = tmp_path / "nested" / "out"

    result = run_evolution(
        "def f(): pass",
        EVALUATOR_CODE,
        config=make_config(),
        output_dir=str(output_dir),
        cleanup=False,
    )

    assert output_dir.is_dir()
    assert controller().init_kwargs["output_dir"] == str(output_dir)
    assert result.output_dir == str(output_dir)


def test_run_evolution_default_output_dir_is_relative_to_the_cwd(fake_controller, tmp_path):
    fake_controller.next_result = make_program()

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config(), cleanup=False)

    assert result.output_dir == "openevolve_output"
    assert (tmp_path / "openevolve_output").is_dir()


def test_run_evolution_with_cleanup_leaves_no_temp_output_dir(fake_controller):
    fake_controller.next_result = make_program()

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    assert result.output_dir is None
    temp_output_dir = controller().init_kwargs["output_dir"]
    assert not Path(temp_output_dir).exists()


def test_run_evolution_with_cleanup_false_keeps_generated_files(fake_controller):
    """Real behaviour: cleanup=False leaves the generated files in the temp dir."""
    fake_controller.next_result = make_program()

    run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config(), cleanup=False)

    instance = controller()
    prepared = [
        Path(instance.init_kwargs["initial_program_path"]),
        Path(instance.init_kwargs["evaluation_file"]),
    ]
    try:
        assert all(path.exists() for path in prepared)
        assert all(path.parent == Path(tempfile.gettempdir()) for path in prepared)
    finally:
        for path in prepared:
            path.unlink(missing_ok=True)


def test_run_evolution_cleans_up_after_a_controller_failure(fake_controller):
    fake_controller.next_result = make_program()
    fake_controller.run_raises = RuntimeError("controller exploded")
    with pytest.raises(RuntimeError, match="controller exploded"):
        run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    instance = controller()
    assert not Path(instance.init_kwargs["initial_program_path"]).exists()
    assert not Path(instance.init_kwargs["evaluation_file"]).exists()
    assert not Path(instance.init_kwargs["output_dir"]).exists()


def test_run_evolution_survives_a_failing_unlink(fake_controller, monkeypatch):
    """Temp-file removal is best-effort: an unlink error is swallowed."""
    fake_controller.next_result = make_program()

    def broken_unlink(path):
        raise OSError("cannot unlink")

    monkeypatch.setattr(api_module.os, "unlink", broken_unlink)

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    assert result.best_program is not None
    leftovers = [
        Path(controller().init_kwargs["initial_program_path"]),
        Path(controller().init_kwargs["evaluation_file"]),
    ]
    assert all(path.exists() for path in leftovers)
    monkeypatch.undo()
    shutil.rmtree(controller().init_kwargs["output_dir"], ignore_errors=True)


def test_run_evolution_survives_a_failing_rmtree(fake_controller, monkeypatch):
    """A failing rmtree is swallowed too, leaving the temp dir behind."""
    fake_controller.next_result = make_program()

    def broken_rmtree(path, *args, **kwargs):
        raise OSError("cannot rmtree")

    monkeypatch.setattr("shutil.rmtree", broken_rmtree)

    result = run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config())

    temp_output_dir = controller().init_kwargs["output_dir"]
    assert result.best_program is not None
    assert Path(temp_output_dir).exists()

    monkeypatch.undo()
    shutil.rmtree(temp_output_dir, ignore_errors=True)


def _cleanup_temp_files(instance: FakeController) -> None:
    """Remove the temp files a cleanup=False run deliberately leaves behind."""
    for key in ("initial_program_path", "evaluation_file"):
        path = instance.init_kwargs.get(key)
        if path and Path(path).parent == Path(tempfile.gettempdir()):
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# run_evolution: cascade evaluation auto-detection
# ---------------------------------------------------------------------------


def test_run_evolution_disables_cascade_when_stages_are_absent(fake_controller):
    fake_controller.next_result = make_program()
    config = make_config(cascade_evaluation=True)

    run_evolution("def f(): pass", EVALUATOR_CODE, config=config)

    assert controller().init_kwargs["config"].evaluator.cascade_evaluation is False


def test_run_evolution_keeps_cascade_when_stage_functions_exist(fake_controller):
    fake_controller.next_result = make_program()
    config = make_config(cascade_evaluation=True)
    evaluator_code = """
def evaluate_stage1(program_path):
    return {"combined_score": 0.1}

def evaluate(program_path):
    return evaluate_stage1(program_path)
"""

    run_evolution("def f(): pass", evaluator_code, config=config)

    assert controller().init_kwargs["config"].evaluator.cascade_evaluation is True


def test_run_evolution_leaves_an_already_disabled_cascade_alone(fake_controller):
    fake_controller.next_result = make_program()
    config = make_config(cascade_evaluation=False)

    run_evolution("def f(): pass", EVALUATOR_CODE, config=config)

    assert controller().init_kwargs["config"].evaluator.cascade_evaluation is False


# ---------------------------------------------------------------------------
# run_evolution: configuration and validation errors
# ---------------------------------------------------------------------------


def test_run_evolution_accepts_a_config_object(fake_controller):
    fake_controller.next_result = make_program()
    config = make_config()
    config.num_iterations = 3

    run_evolution("def f(): pass", EVALUATOR_CODE, config=config)

    assert controller().init_kwargs["config"] is config


def test_run_evolution_loads_a_yaml_config_file(fake_controller, tmp_path):
    fake_controller.next_result = make_program()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("llm:\n  models:\n    - name: gpt-4\n      api_key: from-yaml\n")

    run_evolution("def f(): pass", EVALUATOR_CODE, config=config_file)

    loaded = controller().init_kwargs["config"]
    assert isinstance(loaded, Config)
    assert [model.name for model in loaded.llm.models] == ["gpt-4"]


def test_run_evolution_rejects_a_config_without_llm_models():
    with pytest.raises(ValueError) as excinfo:
        run_evolution("def f(): pass", EVALUATOR_CODE, config=Config())

    assert "No LLM models configured" in str(excinfo.value)


def test_run_evolution_default_config_has_no_llm_models():
    """config=None builds a bare Config(), which fails the same validation."""
    with pytest.raises(ValueError, match="No LLM models configured"):
        run_evolution("def f(): pass", EVALUATOR_CODE)


def test_run_evolution_missing_config_file_silently_falls_back_to_defaults():
    """Real behaviour: load_config() ignores a non-existent path."""
    with pytest.raises(ValueError, match="No LLM models configured"):
        run_evolution("def f(): pass", EVALUATOR_CODE, config="/definitely/not/here.yaml")


def test_run_evolution_rejects_an_invalid_config_file(tmp_path):
    from dacite.exceptions import WrongTypeError

    config_file = tmp_path / "bad.yaml"
    config_file.write_text("llm:\n  models: not-a-list\n")

    with pytest.raises(WrongTypeError, match="llm.models"):
        run_evolution("def f(): pass", EVALUATOR_CODE, config=config_file)


def test_run_evolution_rejects_evaluator_source_without_an_evaluate_function():
    with pytest.raises(ValueError, match="must contain an 'evaluate"):
        run_evolution("def f(): pass", "def not_an_evaluator(path): pass", config=make_config())


@pytest.mark.parametrize("bad_evaluator", [None, 123, 4.5, {"evaluate": "nope"}])
def test_run_evolution_rejects_a_non_callable_evaluator(fake_controller, bad_evaluator):
    fake_controller.next_result = make_program()

    with pytest.raises(ValueError, match="must contain an 'evaluate"):
        run_evolution("def f(): pass", bad_evaluator, config=make_config())


def test_run_evolution_validates_the_evaluator_after_the_config():
    """The LLM-model check runs first, so a bad config wins over a bad evaluator."""
    with pytest.raises(ValueError, match="No LLM models configured"):
        run_evolution("def f(): pass", None, config=Config())


# ---------------------------------------------------------------------------
# evolve_code
# ---------------------------------------------------------------------------


def test_evolve_code_forwards_its_arguments():
    evaluator = lambda program_path: {"combined_score": 1.0}  # noqa: E731
    sentinel = EvolutionResult(None, 1.0, "", {}, None)
    with mock.patch.object(api_module, "run_evolution", return_value=sentinel) as mocked:
        result = evolve_code("def f(): pass", evaluator, iterations=3, output_dir="/tmp/x")

    assert result is sentinel
    assert mocked.call_args.kwargs["initial_program"] == "def f(): pass"
    assert mocked.call_args.kwargs["evaluator"] is evaluator
    assert mocked.call_args.kwargs["iterations"] == 3
    assert mocked.call_args.kwargs["output_dir"] == "/tmp/x"


def test_evolve_code_rejects_a_duplicate_initial_program_kwarg():
    with pytest.raises(TypeError, match="multiple values"):
        evolve_code("def f(): pass", EVALUATOR_CODE, iterations=1, initial_program="other")


def test_evolve_code_end_to_end_uses_the_given_evaluator_source(fake_controller):
    fake_controller.next_result = make_program(code="def f():\n    return 9\n")

    result = evolve_code("def f():\n    return 1\n", EVALUATOR_CODE, config=make_config())

    assert result.best_code == "def f():\n    return 9\n"
    assert Path(controller().init_kwargs["evaluation_file"]).exists() is False  # cleaned up


def test_evolve_code_serializes_a_lambda_evaluator(fake_controller, tmp_path):
    fake_controller.next_result = make_program()

    result = evolve_code(
        "def f():\n    return 1\n",
        lambda program_path: {"combined_score": 0.42},  # noqa: E731
        config=make_config(),
        output_dir=str(tmp_path / "out"),
        cleanup=False,
    )

    assert result.output_dir == str(tmp_path / "out")
    generated = Path(controller().init_kwargs["evaluation_file"]).read_text()
    assert '_user_evaluator = lambda program_path: {"combined_score": 0.42}' in generated


def test_evolve_code_reports_a_bad_evaluator_string(fake_controller):
    with pytest.raises(ValueError, match="must contain an 'evaluate"):
        evolve_code("def f(): pass", "def other(path): pass", config=make_config())


# ---------------------------------------------------------------------------
# evolve_function
# ---------------------------------------------------------------------------


def test_evolve_function_inserts_markers_and_forwards_arguments():
    sentinel = EvolutionResult(None, 1.0, "", {}, None)
    test_cases = [([3, 1, 2], [1, 2, 3])]

    with mock.patch.object(api_module, "run_evolution", return_value=sentinel) as mocked:
        result = evolve_function(bubble_sort, test_cases, iterations=5, cleanup=False)

    assert result is sentinel
    program_source = mocked.call_args.kwargs["initial_program"]
    assert program_source.count("EVOLVE-BLOCK-START") == 1
    assert program_source.count("EVOLVE-BLOCK-END") == 1
    assert "def bubble_sort(arr):" in program_source
    # Markers belong inside the function, after the def line.
    assert program_source.splitlines()[1].strip() == "# EVOLVE-BLOCK-START"
    assert mocked.call_args.kwargs["iterations"] == 5
    assert mocked.call_args.kwargs["cleanup"] is False
    evaluator_code = mocked.call_args.kwargs["evaluator"]
    assert "FUNC_NAME = 'bubble_sort'" in evaluator_code
    assert repr(test_cases) in evaluator_code


def test_evolve_function_keeps_existing_markers():
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_function(with_markers, [(1, 2)], iterations=1)

    program_source = mocked.call_args.kwargs["initial_program"]
    assert program_source.count("EVOLVE-BLOCK-START") == 1
    assert program_source.count("EVOLVE-BLOCK-END") == 1


def test_evolve_function_output_is_valid_python():
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_function(bubble_sort, [([2, 1], [1, 2])], iterations=1)

    compile(mocked.call_args.kwargs["initial_program"], "<program>", "exec")


def test_evolve_function_generated_evaluator_scores_a_correct_program(tmp_path):
    test_cases = [([3, 1, 2], [1, 2, 3]), ([5, 2], [2, 5])]
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_function(bubble_sort, test_cases, iterations=1)
    evaluate = exec_evaluator(mocked.call_args.kwargs["evaluator"])
    program = write_program(tmp_path / "good.py", "def bubble_sort(arr):\n    return sorted(arr)\n")

    metrics = evaluate(program)

    assert metrics["combined_score"] == 1.0
    assert metrics["test_pass_rate"] == 1.0
    assert metrics["tests_passed"] == 2
    assert metrics["total_tests"] == 2
    assert metrics["errors"] == []


def test_evolve_function_generated_evaluator_reports_failures(tmp_path):
    test_cases = [([1], [2]), ([2], [3]), ([3], [4]), ([4], [5]), ([5], [6])]
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_function(bubble_sort, test_cases, iterations=1)
    evaluate = exec_evaluator(mocked.call_args.kwargs["evaluator"])
    program = write_program(tmp_path / "wrong.py", "def bubble_sort(arr):\n    return [0]\n")

    metrics = evaluate(program)

    assert metrics["combined_score"] == 0.0
    assert metrics["tests_passed"] == 0
    assert len(metrics["errors"]) == 3  # truncated to the first three failures
    assert "expected" in metrics["errors"][0]


def test_evolve_function_generated_evaluator_handles_broken_programs(tmp_path):
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_function(bubble_sort, [([1], [1])], iterations=1)
    evaluate = exec_evaluator(mocked.call_args.kwargs["evaluator"])

    broken = write_program(tmp_path / "broken.py", "def bubble_sort(:\n")
    missing = write_program(tmp_path / "missing.py", "def other(arr):\n    return arr\n")

    assert "Failed to execute program" in evaluate(broken)["error"]
    assert evaluate(broken)["combined_score"] == 0.0
    assert "not found" in evaluate(missing)["error"]
    assert evaluate(missing)["combined_score"] == 0.0


def test_evolve_function_rejects_a_builtin():
    with pytest.raises(TypeError, match="builtin"):
        evolve_function(len, [([1], 1)], iterations=1)


def test_evolve_function_rejects_a_lambda():
    """Real behaviour: a lambda has no ``def`` line, so the marker search raises.

    ``StopIteration`` escapes from the generator expression in
    ``openevolve/api.py`` and is a rather opaque error for a user.
    """
    candidate = lambda arr: sorted(arr)  # noqa: E731

    with pytest.raises(StopIteration):
        evolve_function(candidate, [([1], [1])], iterations=1)


def test_evolve_function_rejects_a_duplicate_initial_program_kwarg():
    with pytest.raises(TypeError, match="multiple values"):
        evolve_function(bubble_sort, [([1], [1])], iterations=1, initial_program="other")


# ---------------------------------------------------------------------------
# evolve_algorithm
# ---------------------------------------------------------------------------


def test_evolve_algorithm_inserts_markers_and_embeds_the_benchmark():
    sentinel = EvolutionResult(None, 1.0, "", {}, None)
    with mock.patch.object(api_module, "run_evolution", return_value=sentinel) as mocked:
        result = evolve_algorithm(SumAlgorithm, benchmark_sum, iterations=4)

    assert result is sentinel
    program_source = mocked.call_args.kwargs["initial_program"]
    assert "class SumAlgorithm:" in program_source
    assert program_source.count("EVOLVE-BLOCK-START") == 1
    assert program_source.count("EVOLVE-BLOCK-END") == 1
    assert program_source.splitlines()[1].strip() == "# EVOLVE-BLOCK-START"
    evaluator_code = mocked.call_args.kwargs["evaluator"]
    assert "CLASS_NAME = 'SumAlgorithm'" in evaluator_code
    assert "def benchmark_sum(instance):" in evaluator_code
    assert "benchmark_sum(instance)" in evaluator_code
    assert mocked.call_args.kwargs["iterations"] == 4


def test_evolve_algorithm_keeps_existing_markers():
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_algorithm(MarkedAlgorithm, benchmark_sum, iterations=1)

    program_source = mocked.call_args.kwargs["initial_program"]
    assert program_source.count("EVOLVE-BLOCK-START") == 1
    assert program_source.count("EVOLVE-BLOCK-END") == 1


def test_evolve_algorithm_generated_evaluator_runs_the_benchmark(tmp_path):
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_algorithm(SumAlgorithm, benchmark_sum, iterations=1)
    evaluate = exec_evaluator(mocked.call_args.kwargs["evaluator"])
    program = write_program(
        tmp_path / "alg.py", "class SumAlgorithm:\n    def process(self, data):\n        return 6\n"
    )

    metrics = evaluate(program)

    assert metrics["combined_score"] == 1.0
    assert metrics["score"] == 1.0
    assert metrics["runtime"] == 0.5


def test_evolve_algorithm_generated_evaluator_wraps_a_scalar_result(tmp_path):
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_algorithm(SumAlgorithm, benchmark_scalar, iterations=1)
    evaluate = exec_evaluator(mocked.call_args.kwargs["evaluator"])
    program = write_program(
        tmp_path / "alg.py", "class SumAlgorithm:\n    def process(self, data):\n        return 0\n"
    )

    assert evaluate(program) == {"score": 0.25, "combined_score": 0.25}


def test_evolve_algorithm_generated_evaluator_keeps_a_combined_score(tmp_path):
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_algorithm(SumAlgorithm, benchmark_with_combined, iterations=1)
    evaluate = exec_evaluator(mocked.call_args.kwargs["evaluator"])
    program = write_program(
        tmp_path / "alg.py", "class SumAlgorithm:\n    def process(self, data):\n        return 0\n"
    )

    assert evaluate(program) == {"combined_score": 0.75}


def test_evolve_algorithm_generated_evaluator_reports_a_missing_class(tmp_path):
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_algorithm(SumAlgorithm, benchmark_sum, iterations=1)
    evaluate = exec_evaluator(mocked.call_args.kwargs["evaluator"])
    program = write_program(tmp_path / "alg.py", "class Other:\n    pass\n")

    metrics = evaluate(program)

    assert metrics["combined_score"] == 0.0
    assert "Class 'SumAlgorithm' not found" in metrics["error"]


def test_evolve_algorithm_generated_evaluator_reports_a_broken_class(tmp_path):
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_algorithm(SumAlgorithm, benchmark_sum, iterations=1)
    evaluate = exec_evaluator(mocked.call_args.kwargs["evaluator"])
    program = write_program(
        tmp_path / "alg.py",
        "class SumAlgorithm:\n    def __init__(self):\n        raise ValueError('bad ctor')\n",
    )

    metrics = evaluate(program)

    assert metrics["combined_score"] == 0.0
    assert "bad ctor" in metrics["error"]


def test_evolve_algorithm_rejects_a_builtin_class():
    with pytest.raises(TypeError, match="built-in class"):
        evolve_algorithm(dict, benchmark_sum, iterations=1)


def test_evolve_algorithm_with_a_lambda_benchmark_generates_broken_code():
    """A lambda benchmark is embedded by *name*, which is not valid Python.

    ``benchmark.__name__`` is ``<lambda>``, so the generated evaluator contains
    ``metrics = <lambda>(instance)`` and fails to compile. ``evolve_algorithm``
    reports no error at this point, so the failure only surfaces later.
    """
    with mock.patch.object(api_module, "run_evolution", return_value=None) as mocked:
        evolve_algorithm(SumAlgorithm, lambda instance: {"score": 1.0}, iterations=1)

    evaluator_code = mocked.call_args.kwargs["evaluator"]
    assert "metrics = <lambda>(instance)" in evaluator_code
    with pytest.raises(SyntaxError):
        compile(evaluator_code, "<generated evaluator>", "exec")


# ---------------------------------------------------------------------------
# private helpers used by the public API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("evaluator=lambda p: {'score': 0.8},  # comment", "lambda p: {'score': 0.8}"),
        ("x = lambda p: (1, 2)  # trailing", "lambda p: (1, 2)"),
        ("g = lambda p: {'a': 1}\nother = 2", "lambda p: {'a': 1}"),
        ("h = lambda p: None)", "lambda p: None"),
        ("f = lambda p: 'a,b['", "lambda p: 'a,b['"),
        ("i = lambda: 1", "lambda: 1"),
        # Quirk: the scan looks for the literal text "lambda" anywhere, so prose
        # containing that word is "extracted" too.
        ("no lambda at all", "lambda at all"),
        ("f(,)", None),
    ],
)
def test_extract_lambda_source(source, expected):
    assert _extract_lambda_source(source) == expected


def test_prepare_program_without_a_temp_dir_uses_the_system_temp_dir(tmp_path):
    temp_files: List[str] = []

    path = _prepare_program("def f(): pass", None, temp_files)

    try:
        assert Path(path).parent == Path(tempfile.gettempdir())
        assert temp_files == [path]
        assert "EVOLVE-BLOCK-START" in Path(path).read_text()
    finally:
        Path(path).unlink(missing_ok=True)


def test_prepare_evaluator_without_a_temp_dir_uses_the_system_temp_dir():
    temp_files: List[str] = []

    path = _prepare_evaluator(EVALUATOR_CODE, None, temp_files)

    try:
        assert Path(path).parent == Path(tempfile.gettempdir())
        assert Path(path).read_text() == EVALUATOR_CODE
        assert temp_files == [path]
    finally:
        Path(path).unlink(missing_ok=True)


class CallableEvaluator:
    """Callable object without retrievable source, exercising the fallback path."""

    def __call__(self, program_path):
        return {"combined_score": 0.125}


def test_prepare_evaluator_falls_back_to_the_api_module_globals(tmp_path):
    temp_files: List[str] = []
    before = set(vars(api_module))

    path = _prepare_evaluator(CallableEvaluator(), str(tmp_path), temp_files)

    try:
        source = Path(path).read_text()
        assert "import openevolve.api as api_module" in source
        registered = set(vars(api_module)) - before
        assert len(registered) == 1
        assert registered.pop() in source
        assert import_file(path).evaluate("/tmp/program.py") == {"combined_score": 0.125}
    finally:
        for attribute in set(vars(api_module)) - before:
            delattr(api_module, attribute)


def test_prepare_evaluator_serializes_a_lambda(tmp_path):
    temp_files: List[str] = []

    path = _prepare_evaluator(lambda program_path: {"score": 1.0}, str(tmp_path), temp_files)

    source = Path(path).read_text()
    assert '_user_evaluator = lambda program_path: {"score": 1.0}' in source
    assert import_file(path).evaluate("/tmp/program.py") == {"score": 1.0}
    assert temp_files == [path]


def test_prepare_evaluator_serializes_a_plain_function(tmp_path):
    temp_files: List[str] = []

    path = _prepare_evaluator(benchmark_sum, str(tmp_path), temp_files)

    source = Path(path).read_text()
    assert "def benchmark_sum(instance):" in source
    assert "return benchmark_sum(program_path)" in source


def test_prepare_evaluator_rejects_unserializable_evaluator_code(tmp_path):
    with pytest.raises(ValueError, match="must contain an 'evaluate"):
        _prepare_evaluator("print('no evaluate here')", str(tmp_path), [])


def test_prepare_program_accepts_a_string_path_to_an_existing_file(tmp_path):
    program_file = write_program(tmp_path / "program.py", "def f(): pass\n")
    temp_files: List[str] = []

    assert _prepare_program(program_file, str(tmp_path), temp_files) == program_file
    assert temp_files == []


def test_generated_evaluator_module_is_importable(fake_controller):
    """A generated evaluator must be a valid standalone module."""
    fake_controller.next_result = make_program()

    run_evolution("def f(): pass", EVALUATOR_CODE, config=make_config(), cleanup=False)

    module = import_file(controller().init_kwargs["evaluation_file"])
    assert module.evaluate("ignored") == {"combined_score": 0.9, "score": 0.9}
