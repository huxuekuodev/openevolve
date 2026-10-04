"""Exhaustive edge-case tests for ``openevolve/utils/metrics_utils.py``.

The module exists to survive *mixed* metrics dictionaries (numbers, bools,
strings, containers, NaN, infinities), so most tests here are edge cases rather
than happy paths. Where the module's behaviour is surprising, the test asserts
the real behaviour and says so in a comment.
"""

import math

import pytest

from openevolve.utils.metrics_utils import (
    format_feature_coordinates,
    get_fitness_score,
    safe_numeric_average,
    safe_numeric_sum,
)

# ---------------------------------------------------------------------------
# safe_numeric_average
# ---------------------------------------------------------------------------


def test_safe_numeric_average_of_empty_metrics_is_zero():
    assert safe_numeric_average({}) == 0.0


@pytest.mark.parametrize(
    "metrics,expected",
    [
        ({"a": 1, "b": 2}, 1.5),
        ({"a": 0.0}, 0.0),
        ({"a": -1.0, "b": 3.0}, 1.0),
        ({"a": 1, "b": 2, "c": 3, "d": 4}, 2.5),
        ({"only": 0.75}, 0.75),
    ],
)
def test_safe_numeric_average_of_numbers(metrics, expected):
    assert safe_numeric_average(metrics) == pytest.approx(expected)


@pytest.mark.parametrize(
    "metrics",
    [
        {"note": "great"},
        {"none": None},
        {"tags": ["a", "b"]},
        {"meta": {"k": 1}},
        {"numeric_string": "1"},
        {"complex": complex(1, 2)},
        {"bytes": b"1"},
    ],
)
def test_safe_numeric_average_ignores_non_numeric_values(metrics):
    assert safe_numeric_average(metrics) == 0.0


def test_safe_numeric_average_ignores_non_numeric_values_among_numbers():
    metrics = {
        "score": 0.8,
        "note": "great",
        "tags": ["x"],
        "meta": {"k": 1},
        "none": None,
    }

    assert safe_numeric_average(metrics) == pytest.approx(0.8)


@pytest.mark.parametrize("value", [True, False])
def test_safe_numeric_average_excludes_a_lone_bool(value):
    """bool is a subclass of int, but a flag is not a score."""
    assert safe_numeric_average({"flag": value}) == 0.0


def test_safe_numeric_average_excludes_bools_from_a_mixed_dict():
    # Without the bool guard this would be (0.4 + 1.0) / 2 = 0.7.
    assert safe_numeric_average({"score": 0.4, "valid": True}) == pytest.approx(0.4)
    assert safe_numeric_average({"error": 0.0, "timeout": True}) == 0.0


def test_safe_numeric_average_skips_nan_values():
    assert safe_numeric_average({"a": float("nan")}) == 0.0
    assert safe_numeric_average({"a": float("nan"), "b": 2.0}) == pytest.approx(2.0)
    assert safe_numeric_average({"a": float("nan"), "b": float("nan")}) == 0.0


def test_safe_numeric_average_keeps_infinities():
    assert math.isinf(safe_numeric_average({"a": float("inf")}))
    assert safe_numeric_average({"a": float("-inf")}) == float("-inf")
    assert math.isinf(safe_numeric_average({"a": float("inf"), "b": 1.0}))


def test_safe_numeric_average_skips_values_that_overflow_float():
    """float(10**400) raises OverflowError; the value is skipped, not fatal."""
    assert safe_numeric_average({"huge": 10**400}) == 0.0
    assert safe_numeric_average({"huge": 10**400, "ok": 1.0}) == pytest.approx(1.0)


def test_safe_numeric_average_of_numpy_scalars():
    np = pytest.importorskip("numpy")

    # np.float64 subclasses float -> included; np.int64 and np.bool_ do not -> excluded.
    assert safe_numeric_average({"a": np.float64(2.0), "b": np.int64(2)}) == pytest.approx(2.0)
    assert safe_numeric_average({"a": np.bool_(True), "b": 1.0}) == pytest.approx(1.0)
    assert safe_numeric_average({"a": np.nan, "b": 2.0}) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# safe_numeric_sum
# ---------------------------------------------------------------------------


def test_safe_numeric_sum_of_empty_metrics_is_zero():
    assert safe_numeric_sum({}) == 0.0


@pytest.mark.parametrize(
    "metrics,expected",
    [
        ({"a": 1, "b": 2}, 3.0),
        ({"a": -1.0, "b": 0.5}, -0.5),
        ({"a": 0.1, "b": 0.2}, 0.30000000000000004),
        ({"only": 2}, 2.0),
        ({"note": "x", "none": None, "tags": [1]}, 0.0),
        ({"a": 1.0, "note": "x"}, 1.0),
        ({"a": float("nan"), "b": 2.0}, 2.0),
        ({"a": float("nan")}, 0.0),
        ({"huge": 10**400, "a": 1.0}, 1.0),
    ],
)
def test_safe_numeric_sum(metrics, expected):
    assert safe_numeric_sum(metrics) == pytest.approx(expected)


def test_safe_numeric_sum_excludes_bools_like_the_average_does():
    """
    `safe_numeric_sum` used to count boolean flags while `safe_numeric_average`
    excluded them, so the two siblings disagreed on the same dict.
    """
    assert safe_numeric_sum({"a": True, "b": False}) == 0.0
    assert safe_numeric_sum({"error": 0.0, "timeout": True}) == 0.0
    assert safe_numeric_average({"error": 0.0, "timeout": True}) == 0.0


def test_safe_numeric_sum_keeps_non_boolean_numbers_alongside_flags():
    assert safe_numeric_sum({"a": 1.5, "b": 2, "timeout": True}) == pytest.approx(3.5)


def test_safe_numeric_sum_keeps_infinities():
    assert math.isinf(safe_numeric_sum({"a": float("inf"), "b": -1.0}))


# ---------------------------------------------------------------------------
# get_fitness_score
# ---------------------------------------------------------------------------


def test_get_fitness_score_of_empty_metrics_is_zero():
    assert get_fitness_score({}) == 0.0


def test_get_fitness_score_prefers_combined_score():
    assert get_fitness_score({"combined_score": 0.9, "other": 0.1}) == pytest.approx(0.9)
    assert get_fitness_score({"combined_score": 0.9, "features": "x"}) == pytest.approx(0.9)


def test_get_fitness_score_accepts_an_int_combined_score():
    assert get_fitness_score({"combined_score": 1}) == pytest.approx(1.0)


def test_get_fitness_score_coerces_a_numeric_string_combined_score():
    """float("0.6") works, so a stringified score is accepted as-is."""
    assert get_fitness_score({"combined_score": "0.6"}) == pytest.approx(0.6)


def test_get_fitness_score_falls_back_when_combined_score_is_unusable():
    assert get_fitness_score({"combined_score": "abc", "other": 0.4}) == pytest.approx(0.4)
    assert get_fitness_score({"combined_score": None, "other": 0.4}) == pytest.approx(0.4)
    assert get_fitness_score({"combined_score": [0.5], "other": 0.4}) == pytest.approx(0.4)


def test_get_fitness_score_honours_a_boolean_combined_score():
    """Real behaviour: the combined_score shortcut has no bool guard."""
    assert get_fitness_score({"combined_score": True}) == pytest.approx(1.0)
    assert get_fitness_score({"combined_score": False}) == 0.0


def test_get_fitness_score_averages_all_numeric_metrics_without_combined_score():
    assert get_fitness_score({"a": 0.8, "b": 0.6}) == pytest.approx(0.7)


def test_get_fitness_score_excludes_feature_dimensions():
    metrics = {"score": 0.8, "complexity": 100.0}

    assert get_fitness_score(metrics, ["complexity"]) == pytest.approx(0.8)


def test_get_fitness_score_excludes_several_feature_dimensions():
    metrics = {"score": 0.4, "runtime": 0.6, "lines": 100.0, "depth": 7.0}

    assert get_fitness_score(metrics, ["lines", "depth"]) == pytest.approx(0.5)
    assert get_fitness_score(metrics, ["lines", "depth", "unknown"]) == pytest.approx(0.5)


def test_get_fitness_score_with_none_feature_dimensions_keeps_everything():
    metrics = {"score": 0.8, "complexity": 0.2}

    assert get_fitness_score(metrics, None) == pytest.approx(0.5)
    assert get_fitness_score(metrics) == pytest.approx(0.5)


def test_get_fitness_score_falls_back_when_every_metric_is_a_feature():
    """Backward compatibility: the fallback averages *all* metrics, features included."""
    metrics = {"feat_a": 1.0, "feat_b": 0.0}

    assert get_fitness_score(metrics, ["feat_a", "feat_b"]) == pytest.approx(0.5)


def test_get_fitness_score_fallback_lets_a_feature_back_into_the_average():
    """Real behaviour: the fallback ignores feature_dimensions entirely."""
    metrics = {"feat_a": 1.0, "flag": True}

    assert get_fitness_score(metrics, ["feat_a"]) == pytest.approx(1.0)


def test_get_fitness_score_ignores_bools_when_no_combined_score():
    assert get_fitness_score({"valid": True, "timeout": False}) == 0.0
    assert get_fitness_score({"score": 0.4, "valid": True}) == pytest.approx(0.4)


def test_get_fitness_score_ignores_nan_and_non_numeric_metrics():
    metrics = {"a": float("nan"), "b": 2.0, "note": "x", "none": None}

    assert get_fitness_score(metrics) == pytest.approx(2.0)
    assert get_fitness_score({"a": float("nan"), "note": "x"}) == 0.0


def test_get_fitness_score_skips_values_that_overflow_float():
    """float(10**400) raises OverflowError inside the fitness loop too."""
    assert get_fitness_score({"huge": 10**400, "score": 0.5}) == pytest.approx(0.5)
    assert get_fitness_score({"huge": 10**400}) == 0.0


def test_get_fitness_score_of_only_strings_is_zero():
    assert get_fitness_score({"note": "x", "label": "y"}) == 0.0


# ---------------------------------------------------------------------------
# format_feature_coordinates
# ---------------------------------------------------------------------------


def test_format_feature_coordinates_formats_numbers_with_two_decimals():
    metrics = {"complexity": 10, "diversity": 2.5}

    assert (
        format_feature_coordinates(metrics, ["complexity", "diversity"])
        == "complexity=10.00, diversity=2.50"
    )


def test_format_feature_coordinates_follows_the_dimension_order():
    metrics = {"a": 1.0, "b": 2.0}

    assert format_feature_coordinates(metrics, ["b", "a"]) == "b=2.00, a=1.00"


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.2345, "x=1.23"),
        (1.239, "x=1.24"),
        (-0.005, "x=-0.01"),
        (0, "x=0.00"),
        (3, "x=3.00"),
    ],
)
def test_format_feature_coordinates_rounds_to_two_decimals(value, expected):
    assert format_feature_coordinates({"x": value}, ["x"]) == expected


def test_format_feature_coordinates_renders_non_numeric_values_verbatim():
    metrics = {"mode": "fast", "tags": ["a", "b"], "meta": {"k": 1}, "nothing": None}

    assert (
        format_feature_coordinates(metrics, ["mode", "tags", "meta", "nothing"])
        == "mode=fast, tags=['a', 'b'], meta={'k': 1}, nothing=None"
    )


def test_format_feature_coordinates_skips_missing_and_nan_dimensions():
    metrics = {"present": 1.0, "nan": float("nan")}

    assert format_feature_coordinates(metrics, ["present", "missing", "nan"]) == "present=1.00"


def test_format_feature_coordinates_returns_empty_string_when_nothing_matches():
    assert format_feature_coordinates({}, ["a", "b"]) == ""
    assert format_feature_coordinates({"a": 1.0}, []) == ""
    assert format_feature_coordinates({"a": 1.0}, ["b", "c"]) == ""
    assert format_feature_coordinates({"a": float("nan")}, ["a"]) == ""


def test_format_feature_coordinates_renders_bools_as_numbers():
    """Real behaviour: unlike the metrics formatters, bools become 1.00 / 0.00 here."""
    assert format_feature_coordinates({"flag": True, "other": False}, ["flag", "other"]) == (
        "flag=1.00, other=0.00"
    )


def test_format_feature_coordinates_keeps_infinities():
    assert format_feature_coordinates({"x": float("inf")}, ["x"]) == "x=inf"


def test_format_feature_coordinates_falls_back_for_float_overflow():
    """A value too large for float() is rendered raw rather than dropped."""
    huge = 10**400

    assert format_feature_coordinates({"x": huge}, ["x"]) == f"x={huge}"
