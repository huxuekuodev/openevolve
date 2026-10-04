"""
Tests for the ``openevolve-run`` command-line interface.

``openevolve/cli.py`` was the only module in the package with no test coverage
at all, even though it is the entry point advertised in the README. These tests
drive ``main_async()`` the way a user would (through ``sys.argv``) and assert on
its exit code and output.

``OpenEvolve`` itself is replaced with a fake: the evolution loop is covered by
the controller/database/process tests, while what matters here is the argument
handling, the pre-flight validation, and the reporting/exit-code contract.
"""

import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openevolve import cli
from openevolve.config import Config


class _FakeOpenEvolve:
    """Stand-in for ``openevolve.OpenEvolve`` that records how it was called."""

    instances: list["_FakeOpenEvolve"] = []

    # Set by tests to control what run() returns.
    best_program: object = None
    run_error: BaseException | None = None
    run_result: object = None

    def __init__(self, initial_program_path, evaluation_file, config, output_dir=None):
        self.initial_program_path = initial_program_path
        self.evaluation_file = evaluation_file
        self.config = config
        self.output_dir = str(output_dir) if output_dir else str(Path(initial_program_path).parent)
        self.database = SimpleNamespace(last_iteration=0, load=MagicMock())
        self.run_calls: list[dict] = []
        self.run_returned = _FakeOpenEvolve.run_result
        _FakeOpenEvolve.instances.append(self)

    async def run(self, iterations=None, target_score=None, checkpoint_path=None):
        self.run_calls.append(
            {
                "iterations": iterations,
                "target_score": target_score,
                "checkpoint_path": checkpoint_path,
            }
        )
        if _FakeOpenEvolve.run_error is not None:
            raise _FakeOpenEvolve.run_error
        return _FakeOpenEvolve.best_program


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeOpenEvolve.instances = []
    # Default to a successful run; tests that care about the failure paths set
    # this to None or install run_error explicitly.
    _FakeOpenEvolve.best_program = SimpleNamespace(metrics={"combined_score": 1.0})
    _FakeOpenEvolve.run_error = None
    _FakeOpenEvolve.run_result = None
    yield
    _FakeOpenEvolve.instances = []


@pytest.fixture
def files(tmp_path):
    """A minimal initial program and evaluator on disk."""
    program = tmp_path / "initial_program.py"
    program.write_text("# EVOLVE-BLOCK-START\ndef solve(x):\n    return x\n# EVOLVE-BLOCK-END\n")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("def evaluate(program_path):\n    return {'combined_score': 1.0}\n")
    return program, evaluator


def _invoke(argv, patch_openevolve=True):
    """Run ``cli.main_async()`` with ``argv`` and a fake OpenEvolve."""
    with patch.object(sys, "argv", ["openevolve-run", *argv]):
        if not patch_openevolve:
            return asyncio.run(cli.main_async())
        with patch.object(cli, "OpenEvolve", _FakeOpenEvolve):
            return asyncio.run(cli.main_async())


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["openevolve-run", "a.py", "b.py"])
        args = cli.parse_args()

        assert args.initial_program == "a.py"
        assert args.evaluation_file == "b.py"
        assert args.config is None
        assert args.output is None
        assert args.iterations is None
        assert args.target_score is None
        assert args.log_level is None
        assert args.checkpoint is None
        assert args.api_base is None
        assert args.primary_model is None
        assert args.secondary_model is None

    def test_short_flags(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "openevolve-run",
                "a.py",
                "b.py",
                "-c",
                "cfg.yaml",
                "-o",
                "out",
                "-i",
                "7",
                "-t",
                "0.5",
                "-l",
                "DEBUG",
            ],
        )
        args = cli.parse_args()

        assert args.config == "cfg.yaml"
        assert args.output == "out"
        assert args.iterations == 7
        assert args.target_score == 0.5
        assert args.log_level == "DEBUG"

    def test_long_flags(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "openevolve-run",
                "a.py",
                "b.py",
                "--config",
                "cfg.yaml",
                "--iterations",
                "3",
                "--target-score",
                "1.25",
                "--checkpoint",
                "ckpt_10",
                "--api-base",
                "http://localhost:8000/v1",
                "--primary-model",
                "m1",
                "--secondary-model",
                "m2",
            ],
        )
        args = cli.parse_args()

        assert args.iterations == 3
        assert args.target_score == 1.25
        assert args.checkpoint == "ckpt_10"
        assert args.api_base == "http://localhost:8000/v1"
        assert args.primary_model == "m1"
        assert args.secondary_model == "m2"

    def test_invalid_log_level_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["openevolve-run", "a.py", "b.py", "--log-level", "CHATTY"]
        )
        with pytest.raises(SystemExit):
            cli.parse_args()

    def test_non_integer_iterations_are_rejected(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["openevolve-run", "a.py", "b.py", "--iterations", "many"])
        with pytest.raises(SystemExit):
            cli.parse_args()

    def test_positional_arguments_are_required(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["openevolve-run", "only_one.py"])
        with pytest.raises(SystemExit):
            cli.parse_args()


# --------------------------------------------------------------------------
# Pre-flight validation
# --------------------------------------------------------------------------


class TestPreflightValidation:
    def test_missing_initial_program(self, tmp_path, capsys):
        evaluator = tmp_path / "evaluator.py"
        evaluator.write_text("")

        exit_code = _invoke([str(tmp_path / "nope.py"), str(evaluator)])

        assert exit_code == 1
        assert "not found" in capsys.readouterr().out
        assert _FakeOpenEvolve.instances == []

    def test_missing_evaluation_file(self, files, capsys):
        program, _ = files

        exit_code = _invoke([str(program), str(program.parent / "nope.py")])

        assert exit_code == 1
        assert "not found" in capsys.readouterr().out
        assert _FakeOpenEvolve.instances == []

    def test_missing_checkpoint_directory(self, files, capsys):
        program, evaluator = files

        exit_code = _invoke([str(program), str(evaluator), "--checkpoint", "/nope/ckpt_1"])

        assert exit_code == 1
        assert "Checkpoint directory" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------


class TestOverrides:
    def test_iterations_and_target_score_reach_run(self, files):
        program, evaluator = files

        exit_code = _invoke([str(program), str(evaluator), "-i", "12", "-t", "0.9"])

        assert exit_code == 0
        controller = _FakeOpenEvolve.instances[-1]
        assert controller.run_calls == [
            {"iterations": 12, "target_score": 0.9, "checkpoint_path": None}
        ]

    def test_output_directory_override(self, files, tmp_path):
        program, evaluator = files
        out = tmp_path / "custom_out"

        assert _invoke([str(program), str(evaluator), "-o", str(out)]) == 0
        assert _FakeOpenEvolve.instances[-1].output_dir == str(out)

    def test_api_base_override(self, files, capsys):
        program, evaluator = files

        exit_code = _invoke(
            [str(program), str(evaluator), "--api-base", "http://localhost:9999/v1"]
        )

        assert exit_code == 0
        assert "Using API base: http://localhost:9999/v1" in capsys.readouterr().out
        assert _FakeOpenEvolve.instances[-1].config.llm.api_base == "http://localhost:9999/v1"

    def test_model_overrides_rebuild_models(self, files, capsys):
        program, evaluator = files

        exit_code = _invoke(
            [
                str(program),
                str(evaluator),
                "--primary-model",
                "primary-x",
                "--secondary-model",
                "secondary-y",
            ]
        )

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "Using primary model: primary-x" in out
        assert "Using secondary model: secondary-y" in out
        assert "Applied CLI model overrides - active models:" in out
        assert "Model 1: primary-x" in out
        assert "Model 2: secondary-y" in out

        models = _FakeOpenEvolve.instances[-1].config.llm.models
        assert [m.name for m in models] == ["primary-x", "secondary-y"]

    def test_log_level_override(self, files):
        program, evaluator = files
        root = logging.getLogger()
        original_level = root.level
        root.setLevel(logging.WARNING)
        try:
            assert _invoke([str(program), str(evaluator), "-l", "DEBUG"]) == 0
            assert root.level == logging.DEBUG
        finally:
            root.setLevel(original_level)

    def test_config_file_is_loaded(self, files, tmp_path):
        program, evaluator = files
        config_file = tmp_path / "config.yaml"
        config_file.write_text("max_iterations: 42\n")

        assert _invoke([str(program), str(evaluator), "-c", str(config_file)]) == 0
        assert _FakeOpenEvolve.instances[-1].config.max_iterations == 42


# --------------------------------------------------------------------------
# Checkpoint resume
# --------------------------------------------------------------------------


class TestCheckpointResume:
    def test_checkpoint_is_loaded_and_passed_through(self, files, tmp_path, capsys):
        program, evaluator = files
        checkpoint = tmp_path / "checkpoint_50"
        checkpoint.mkdir()

        exit_code = _invoke([str(program), str(evaluator), "--checkpoint", str(checkpoint)])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert f"Loading checkpoint from {checkpoint}" in out
        assert "Checkpoint loaded successfully (iteration 0)" in out
        assert _FakeOpenEvolve.instances[-1].run_calls[0]["checkpoint_path"] == str(checkpoint)


# --------------------------------------------------------------------------
# Reporting and exit codes
# --------------------------------------------------------------------------


class TestReporting:
    def test_best_program_metrics_are_printed(self, files, capsys):
        program, evaluator = files
        _FakeOpenEvolve.best_program = SimpleNamespace(
            metrics={"combined_score": 0.81234, "label": "best"}
        )

        exit_code = _invoke([str(program), str(evaluator)])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "Evolution complete!" in out
        assert "Best program metrics:" in out
        assert "combined_score: 0.8123" in out  # floats are formatted to 4 dp
        assert "label: best" in out  # non-numerics are printed verbatim

    def test_none_best_program_exits_with_error(self, files, capsys):
        program, evaluator = files
        _FakeOpenEvolve.best_program = None

        exit_code = _invoke([str(program), str(evaluator)])

        assert exit_code == 1
        assert "No program was produced" in capsys.readouterr().out

    def test_exception_during_run_returns_one(self, files, capsys):
        program, evaluator = files
        _FakeOpenEvolve.run_error = RuntimeError("boom")

        exit_code = _invoke([str(program), str(evaluator)])

        assert exit_code == 1
        assert "Error: boom" in capsys.readouterr().out

    def test_latest_checkpoint_is_reported(self, files, tmp_path, capsys):
        program, evaluator = files
        out_dir = tmp_path / "out"
        checkpoints = out_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        # Created out of order on purpose: selection must be by numeric suffix.
        for name in ("checkpoint_100", "checkpoint_5", "checkpoint_20"):
            (checkpoints / name).mkdir()
        (checkpoints / "not_a_checkpoint.txt").write_text("ignored")

        exit_code = _invoke([str(program), str(evaluator), "-o", str(out_dir)])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert f"Latest checkpoint saved at: {checkpoints / 'checkpoint_100'}" in out
        assert "To resume, use: --checkpoint" in out

    def test_no_checkpoint_message_without_checkpoints(self, files, tmp_path, capsys):
        program, evaluator = files
        out_dir = tmp_path / "empty_out"

        exit_code = _invoke([str(program), str(evaluator), "-o", str(out_dir)])

        assert exit_code == 0
        assert "Latest checkpoint saved at" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


class TestMain:
    def test_main_returns_the_exit_code_from_main_async(self, files):
        program, evaluator = files

        with patch.object(sys, "argv", ["openevolve-run", str(program), str(evaluator)]):
            with patch.object(cli, "OpenEvolve", _FakeOpenEvolve):
                assert cli.main() == 0

    def test_console_script_target_is_importable(self):
        """`openevolve-run` is declared as an entry point in pyproject.toml."""
        import importlib

        module = importlib.import_module("openevolve.cli")
        assert callable(module.main)
