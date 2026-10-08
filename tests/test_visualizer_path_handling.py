"""
Regression tests for how the visualizer resolves its --path argument.

`find_latest_checkpoint` first asks "is the given path *itself* a checkpoint
directory?" by looking at `os.path.basename(path).startswith("checkpoint_")`.
Shell tab-completion appends a trailing separator to directories, and
`os.path.basename(".../checkpoint_50/")` is `""` rather than `"checkpoint_50"`,
so a perfectly valid `--path .../checkpoints/checkpoint_50/` fell through to a
glob that searched *inside* the checkpoint and found nothing. The server then
served an empty tree with no error, which is how the bug reached a user.

The documented usage (`--help` and the README both say `checkpoints/checkpoint_*`)
therefore only worked without a trailing slash.
"""

import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import flask
except ImportError:  # pragma: no cover - visualizer extras not installed
    flask = None

VISUALIZER = Path(__file__).resolve().parent.parent / "scripts" / "visualizer.py"


def _load_visualizer():
    """Execute visualizer.py as a module (without its __main__ block)."""
    with patch.object(sys, "path", [str(VISUALIZER.parent), *sys.path]):
        return runpy.run_path(str(VISUALIZER))


def _write_checkpoint(ckpt_dir, island_ids, mtime):
    """Create a minimal on-disk checkpoint that load_evolution_data accepts."""
    programs = ckpt_dir / "programs"
    programs.mkdir(parents=True, exist_ok=True)
    islands = []
    for pid in island_ids:
        (programs / f"{pid}.json").write_text(
            json.dumps(
                {
                    "id": pid,
                    "code": f"def f():\n    return {pid!r}\n",
                    "metrics": {"combined_score": 0.5},
                    "language": "python",
                    # Program.to_dict() always emits metadata; the sanitizer reads it.
                    "metadata": {},
                }
            )
        )
    islands.append(list(island_ids))
    (ckpt_dir / "metadata.json").write_text(json.dumps({"islands": islands}))
    os.utime(ckpt_dir, (mtime, mtime))
    return ckpt_dir


@unittest.skipIf(flask is None, "flask is not installed")
class TestFindLatestCheckpoint(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "openevolve_output"
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir(parents=True)
        self.older = _write_checkpoint(checkpoints / "checkpoint_5", ["a"], mtime=1_000_000)
        self.newer = _write_checkpoint(checkpoints / "checkpoint_10", ["b", "c"], mtime=2_000_000)
        self.find = _load_visualizer()["find_latest_checkpoint"]

    def tearDown(self):
        self._tmp.cleanup()

    # --- the regression -------------------------------------------------

    def test_checkpoint_path_with_trailing_slash_is_recognised(self):
        """The exact form shell tab-completion produces: `.../checkpoint_10/`."""
        self.assertEqual(self.find(str(self.newer) + os.sep), str(self.newer))

    def test_checkpoint_path_without_trailing_slash_is_recognised(self):
        self.assertEqual(self.find(str(self.newer)), str(self.newer))

    def test_checkpoint_path_tolerates_repeated_separators(self):
        self.assertEqual(self.find(str(self.newer) + os.sep + os.sep), str(self.newer))

    def test_dot_prefixed_checkpoint_path_is_recognised(self):
        self.assertEqual(self.find(os.path.join(".", str(self.newer))), str(self.newer))

    def test_normpath_does_not_break_a_plain_checkpoint_name(self):
        with patch("os.getcwd", return_value=str(self.newer.parent)):
            self.assertEqual(self.find("checkpoint_5" + os.sep), "checkpoint_5")

    # --- discovery from a parent directory -------------------------------

    def test_finds_newest_checkpoint_from_the_output_dir(self):
        self.assertEqual(self.find(str(self.root)), str(self.newer))

    def test_finds_newest_checkpoint_from_the_checkpoints_dir(self):
        self.assertEqual(self.find(str(self.root / "checkpoints")), str(self.newer))

    def test_trailing_slash_on_a_parent_dir_still_discovers(self):
        self.assertEqual(self.find(str(self.root) + os.sep), str(self.newer))

    def test_returns_none_when_nothing_is_found(self):
        empty = Path(self._tmp.name) / "empty"
        empty.mkdir()
        self.assertIsNone(self.find(str(empty) + os.sep))


@unittest.skipIf(flask is None, "flask is not installed")
class TestLoadEvolutionData(unittest.TestCase):
    """The recognised checkpoint must actually yield a non-empty tree."""

    def test_checkpoint_reached_with_a_trailing_slash_renders_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _write_checkpoint(Path(tmp) / "checkpoint_3", ["p1", "p2", "p3"], mtime=1)
            ns = _load_visualizer()
            found = ns["find_latest_checkpoint"](str(ckpt) + os.sep)
            data = ns["load_evolution_data"](found)

            self.assertEqual(found, str(ckpt))
            self.assertEqual(len(data["nodes"]), 3)
            self.assertEqual(data["checkpoint_dir"], str(ckpt))


@unittest.skipIf(flask is None, "flask is not installed")
class TestStartupLogging(unittest.TestCase):
    """The evolution tree URL must be shown, and shown as a bare clickable URL."""

    def _run_and_capture(self, *args):
        with (
            patch.object(flask.Flask, "run"),
            patch.object(sys, "argv", ["visualizer.py", "--path", ".", *args]),
            patch.object(sys, "path", [str(VISUALIZER.parent), *sys.path]),
            patch.dict(os.environ),
            self.assertLogs("__main__", level="INFO") as logs,
        ):
            runpy.run_path(str(VISUALIZER), run_name="__main__")
        return "\n".join(logs.output)

    def test_root_url_is_logged_as_a_bare_url(self):
        output = self._run_and_capture("--port", "8123")
        self.assertIn("Evolution tree UI: http://127.0.0.1:8123/", output)

    def test_manual_url_is_labelled_as_optional(self):
        output = self._run_and_capture("--port", "8123")
        self.assertIn("http://127.0.0.1:8123/manual", output)
        # The manual queue is empty unless manual mode was used, so the log must
        # say so rather than presenting it as the main UI.
        self.assertIn("Manual-mode queue", output)


if __name__ == "__main__":
    unittest.main()
