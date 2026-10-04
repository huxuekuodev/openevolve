"""Behavioural tests for ``openevolve/utils/trace_export_utils.py``.

Every test round-trips through the real writers and readers on ``tmp_path``;
nothing is mocked. The HDF5 read/write paths are exercised only when ``h5py``
is importable, but the "h5py is missing" error paths are covered on both
kinds of environment.
"""

import gzip
import json
import time
from typing import Any, Dict, List

import pytest

from openevolve.utils.trace_export_utils import (
    append_trace_jsonl,
    export_traces,
    export_traces_hdf5,
    export_traces_json,
    export_traces_jsonl,
    load_traces,
    load_traces_hdf5,
    load_traces_json,
    load_traces_jsonl,
)


class FakeTrace:
    """Minimal stand-in for an evolution trace: only ``to_dict`` is required."""

    def __init__(self, **data: Any):
        self._data = data

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


def h5py_available() -> bool:
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture
def traces() -> List[FakeTrace]:
    return [
        FakeTrace(iteration=0, score=0.25, code="print(1)"),
        FakeTrace(iteration=1, score=0.75, code="print(2)", nested={"a": [1, 2]}),
    ]


# ---------------------------------------------------------------------------
# export_traces_jsonl / append_trace_jsonl / load_traces_jsonl
# ---------------------------------------------------------------------------


def test_export_traces_jsonl_writes_one_object_per_line(tmp_path, traces):
    path = tmp_path / "traces.jsonl"

    assert export_traces_jsonl(traces, path) is None

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == [trace.to_dict() for trace in traces]
    assert load_traces_jsonl(path) == [trace.to_dict() for trace in traces]


def test_export_traces_jsonl_creates_missing_parent_directories(tmp_path, traces):
    path = tmp_path / "deeply" / "nested" / "traces.jsonl"

    export_traces_jsonl(traces, path)

    assert path.exists()
    assert len(load_traces_jsonl(path)) == 2


def test_export_traces_jsonl_accepts_plain_dicts(tmp_path):
    path = tmp_path / "plain.jsonl"

    export_traces_jsonl([{"a": 1}, {"b": 2}], path)

    assert load_traces_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_export_traces_jsonl_with_empty_list_writes_an_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"

    export_traces_jsonl([], path)

    assert path.exists()
    assert path.read_text() == ""
    assert load_traces_jsonl(path) == []


def test_export_traces_jsonl_compresses_and_appends_the_gz_suffix(tmp_path, traces):
    path = tmp_path / "traces.jsonl"

    export_traces_jsonl(traces, path, compress=True)

    gz_path = tmp_path / "traces.jsonl.gz"
    assert gz_path.exists()
    assert not path.exists()  # the plain path is not created

    with gzip.open(gz_path, "rt") as handle:
        decompressed = handle.read()
    assert [json.loads(line) for line in decompressed.splitlines()] == [
        trace.to_dict() for trace in traces
    ]
    assert load_traces_jsonl(gz_path, compress=True) == [t.to_dict() for t in traces]


def test_export_traces_jsonl_does_not_double_the_gz_suffix(tmp_path, traces):
    path = tmp_path / "traces.gz"

    export_traces_jsonl(traces, path, compress=True)

    assert path.exists()
    assert not (tmp_path / "traces.gz.gz").exists()
    assert load_traces_jsonl(path, compress=True) == [t.to_dict() for t in traces]


def test_load_traces_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "blanks.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"b": 2}\n')

    assert load_traces_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_load_traces_jsonl_detects_the_gz_suffix_without_the_flag(tmp_path, traces):
    """``compress`` is optional for .gz paths: the suffix is enough."""
    path = tmp_path / "traces.jsonl.gz"
    export_traces_jsonl(traces, path, compress=True)

    assert load_traces_jsonl(path) == [t.to_dict() for t in traces]


def test_load_traces_jsonl_needs_the_flag_for_a_gzip_file_without_gz_suffix(tmp_path):
    path = tmp_path / "traces.data"
    with gzip.open(path, "wt") as handle:
        handle.write('{"a": 1}\n')

    with pytest.raises(UnicodeDecodeError):
        load_traces_jsonl(path)

    assert load_traces_jsonl(path, compress=True) == [{"a": 1}]


def test_append_trace_jsonl_accumulates_lines(tmp_path):
    path = tmp_path / "appended.jsonl"

    for i in range(3):
        append_trace_jsonl(FakeTrace(iteration=i), path)

    assert load_traces_jsonl(path) == [{"iteration": 0}, {"iteration": 1}, {"iteration": 2}]


def test_append_trace_jsonl_does_not_truncate_existing_content(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text('{"iteration": "pre-existing"}\n')

    append_trace_jsonl(FakeTrace(iteration=1), path)

    assert load_traces_jsonl(path) == [{"iteration": "pre-existing"}, {"iteration": 1}]


def test_append_trace_jsonl_accepts_plain_dicts(tmp_path):
    path = tmp_path / "dicts.jsonl"

    append_trace_jsonl({"a": 1}, path)
    append_trace_jsonl({"b": 2}, path)

    assert load_traces_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_append_trace_jsonl_compressed_round_trip(tmp_path):
    path = tmp_path / "appended.jsonl"

    append_trace_jsonl(FakeTrace(iteration=0), path, compress=True)
    append_trace_jsonl(FakeTrace(iteration=1), path, compress=True)

    gz_path = tmp_path / "appended.jsonl.gz"
    assert gz_path.exists()
    assert load_traces_jsonl(gz_path, compress=True) == [{"iteration": 0}, {"iteration": 1}]


def test_append_trace_jsonl_does_not_double_a_gz_suffix(tmp_path):
    path = tmp_path / "already.gz"

    append_trace_jsonl(FakeTrace(iteration=0), path, compress=True)

    assert path.exists()
    assert not (tmp_path / "already.gz.gz").exists()
    assert load_traces_jsonl(path) == [{"iteration": 0}]


# ---------------------------------------------------------------------------
# export_traces_json / load_traces_json
# ---------------------------------------------------------------------------


def test_export_traces_json_adds_default_metadata(tmp_path, traces):
    path = tmp_path / "traces.json"
    before = time.time()

    assert export_traces_json(traces, path) is None

    data = json.loads(path.read_text())
    assert data["traces"] == [trace.to_dict() for trace in traces]
    assert data["metadata"]["total_traces"] == 2
    assert before <= data["metadata"]["exported_at"] <= time.time()


def test_export_traces_json_mutates_the_callers_metadata_dict(tmp_path, traces):
    """Documented real behaviour: ``setdefault`` writes into the caller's dict."""
    metadata = {"author": "tester", "total_traces": 99}

    export_traces_json(traces, tmp_path / "traces.json", metadata=metadata)

    assert metadata["author"] == "tester"
    assert metadata["total_traces"] == 99  # setdefault never overwrites
    assert isinstance(metadata["exported_at"], float)

    data = json.loads((tmp_path / "traces.json").read_text())
    assert data["metadata"] == metadata


def test_export_traces_json_leaves_an_empty_metadata_dict_untouched(tmp_path, traces):
    """Quirk: ``metadata or {}`` replaces a *falsy* (empty) dict with a new one."""
    metadata: Dict[str, Any] = {}

    export_traces_json(traces, tmp_path / "traces.json", metadata=metadata)

    assert metadata == {}
    data = json.loads((tmp_path / "traces.json").read_text())
    assert data["metadata"]["total_traces"] == 2


def test_export_traces_json_empty_trace_list(tmp_path):
    path = tmp_path / "empty.json"

    export_traces_json([], path)

    traces, metadata = load_traces_json(path)
    assert traces == []
    assert metadata["total_traces"] == 0


def test_load_traces_json_defaults_for_missing_keys(tmp_path):
    path = tmp_path / "minimal.json"
    path.write_text("{}")

    assert load_traces_json(path) == ([], {})


def test_export_traces_json_round_trips_plain_dicts(tmp_path):
    path = tmp_path / "dicts.json"

    export_traces_json([{"a": 1}], path)

    traces, _ = load_traces_json(path)
    assert traces == [{"a": 1}]


# ---------------------------------------------------------------------------
# export_traces dispatch
# ---------------------------------------------------------------------------


def test_export_traces_jsonl_is_the_default_format(tmp_path, traces):
    path = tmp_path / "default.jsonl"

    export_traces(traces, path)

    assert load_traces_jsonl(path) == [t.to_dict() for t in traces]


def test_export_traces_format_is_case_insensitive(tmp_path, traces):
    path = tmp_path / "upper.JSONL"

    export_traces(traces, path, format="JSONL")

    assert load_traces_jsonl(path) == [t.to_dict() for t in traces]


def test_export_traces_json_route_passes_metadata(tmp_path, traces):
    path = tmp_path / "routed.json"

    export_traces(traces, path, format="json", metadata={"run": "unit-test"})

    traces_back, metadata = load_traces_json(path)
    assert metadata["run"] == "unit-test"
    assert traces_back == [t.to_dict() for t in traces]


def test_export_traces_jsonl_route_honours_compress(tmp_path, traces):
    path = tmp_path / "routed.jsonl"

    export_traces(traces, path, format="jsonl", compress=True)

    assert (tmp_path / "routed.jsonl.gz").exists()


def test_export_traces_ignores_metadata_for_jsonl(tmp_path, traces):
    """``metadata`` is documented as json/hdf5 only; jsonl silently drops it."""
    path = tmp_path / "ignored.jsonl"

    export_traces(traces, path, format="jsonl", metadata={"run": "dropped"})

    assert load_traces_jsonl(path) == [t.to_dict() for t in traces]


@pytest.mark.parametrize("bad_format", ["yaml", "csv", "", "jsonlines"])
def test_export_traces_rejects_unsupported_formats(tmp_path, traces, bad_format):
    with pytest.raises(ValueError) as excinfo:
        export_traces(traces, tmp_path / "out", format=bad_format)

    assert "Unsupported format" in str(excinfo.value)
    assert bad_format in str(excinfo.value)


# ---------------------------------------------------------------------------
# load_traces format detection
# ---------------------------------------------------------------------------


def test_load_traces_detects_jsonl_by_extension(tmp_path, traces):
    path = tmp_path / "traces.jsonl"
    export_traces_jsonl(traces, path)

    assert load_traces(path) == [t.to_dict() for t in traces]


def test_load_traces_detects_gzip_jsonl_and_returns_a_list(tmp_path, traces):
    path = tmp_path / "traces.jsonl.gz"
    export_traces_jsonl(traces, path, compress=True)

    loaded = load_traces(path)

    assert isinstance(loaded, list)
    assert loaded == [t.to_dict() for t in traces]


def test_load_traces_detects_json_by_extension(tmp_path, traces):
    path = tmp_path / "traces.json"
    export_traces_json(traces, path, metadata={"run": "detect"})

    traces_back, metadata = load_traces(path)

    assert traces_back == [t.to_dict() for t in traces]
    assert metadata["run"] == "detect"


def test_load_traces_detects_jsonl_from_content(tmp_path):
    path = tmp_path / "no-extension-jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n')

    assert load_traces(path) == [{"a": 1}, {"b": 2}]


def test_load_traces_detects_json_from_content(tmp_path, traces):
    path = tmp_path / "no-extension-json"
    export_traces_json(traces, path)

    traces_back, metadata = load_traces(path)

    assert traces_back == [t.to_dict() for t in traces]
    assert metadata["total_traces"] == 2


def test_load_traces_detects_indented_json_from_content(tmp_path):
    """Indented JSON has no ``\\n{`` sequence, so the content sniff calls it JSON."""
    path = tmp_path / "indented-data"
    path.write_text(json.dumps({"metadata": {}, "traces": [{"a": 1}]}, indent=2))

    traces, metadata = load_traces(path)

    assert traces == [{"a": 1}]
    assert metadata == {}


def test_load_traces_defaults_to_jsonl_for_an_empty_unknown_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert load_traces(path) == []


def test_load_traces_jsonl_guess_fails_on_non_json_content(tmp_path):
    """The fallback guess is "jsonl", so non-JSON content raises."""
    path = tmp_path / "garbage.bin"
    path.write_text("hello\nworld\n")

    with pytest.raises(json.JSONDecodeError):
        load_traces(path)


def test_load_traces_explicit_format_overrides_detection(tmp_path, traces):
    """An explicit format wins over the (misleading) file extension."""
    path = tmp_path / "traces.json"  # extension says json, content is jsonl
    export_traces_jsonl(traces, path)

    assert load_traces(path, format="jsonl") == [t.to_dict() for t in traces]


def test_load_traces_rejects_unsupported_formats(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("{}\n")

    with pytest.raises(ValueError, match="Unsupported format: yaml"):
        load_traces(path, format="yaml")


def test_load_traces_detects_hdf5_magic_bytes(tmp_path):
    path = tmp_path / "mystery.bin"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 32)

    if h5py_available():
        pytest.skip("h5py installed, so the ImportError path does not apply")

    with pytest.raises(ImportError, match="h5py not installed"):
        load_traces(path)


@pytest.mark.parametrize("suffix", [".h5", ".hdf5"])
def test_load_traces_detects_hdf5_by_extension_requires_h5py(tmp_path, suffix):
    path = tmp_path / f"traces{suffix}"
    path.write_bytes(b"not really hdf5")

    if h5py_available():
        pytest.skip("h5py installed, so the ImportError path does not apply")

    with pytest.raises(ImportError, match="h5py not installed"):
        load_traces(path)


# ---------------------------------------------------------------------------
# HDF5 (optional dependency)
# ---------------------------------------------------------------------------


def test_export_traces_hdf5_requires_h5py(tmp_path, traces):
    if h5py_available():
        pytest.skip("h5py installed, so the ImportError path does not apply")

    with pytest.raises(ImportError, match="h5py not installed"):
        export_traces_hdf5(traces, tmp_path / "traces.h5")


def test_load_traces_hdf5_requires_h5py(tmp_path):
    if h5py_available():
        pytest.skip("h5py installed, so the ImportError path does not apply")

    with pytest.raises(ImportError, match="h5py not installed"):
        load_traces_hdf5(tmp_path / "traces.h5")


def test_export_traces_route_to_hdf5_requires_h5py(tmp_path, traces):
    if h5py_available():
        pytest.skip("h5py installed, so the ImportError path does not apply")

    with pytest.raises(ImportError, match="h5py not installed"):
        export_traces(traces, tmp_path / "traces.h5", format="hdf5")


def test_export_traces_hdf5_round_trip(tmp_path):
    pytest.importorskip("h5py")
    traces = [
        FakeTrace(iteration=0, score=0.5, code="print(1)", tags=["a", "b"], nested={"k": "v"}),
        FakeTrace(iteration=1, score=None, code="print(2)"),
    ]
    path = tmp_path / "traces.h5"

    export_traces_hdf5(traces, path, metadata={"run": "hdf5", "count": 2, "nested": {"a": 1}})

    loaded, metadata = load_traces_hdf5(path)
    assert path.exists()
    assert metadata["run"] == "hdf5"
    assert metadata["count"] == 2
    assert metadata["nested"] == {"a": 1}
    assert metadata["total_traces"] == 2
    assert loaded[0]["code"] == "print(1)"
    assert loaded[0]["score"] == 0.5
    assert loaded[0]["tags"] == ["a", "b"]
    assert loaded[0]["nested"] == {"k": "v"}
    # None values are skipped entirely.
    assert "score" not in loaded[1]


def test_load_traces_detects_hdf5_by_extension(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "traces.hdf5"
    export_traces_hdf5([FakeTrace(iteration=7)], path, metadata={"run": "detect"})

    loaded, metadata = load_traces(path)

    assert metadata["run"] == "detect"
    assert loaded[0]["iteration"] == 7


def test_load_traces_hdf5_rejects_a_non_hdf5_file(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "not-really.h5"
    path.write_text("hello")

    with pytest.raises(OSError):
        load_traces_hdf5(path)
