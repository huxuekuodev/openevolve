"""
Request-parameter routing for OpenAI reasoning models.

Reasoning models (o-series, gpt-5, gpt-oss) reject ``max_tokens``,
``temperature`` and ``top_p`` and require ``max_completion_tokens`` instead.
OpenEvolve decides which set to send from the *model name*, not from the API
base URL, so the same routing must hold for OpenAI, Azure, regional
(``eu.``/``apac.``), OptiLLM and OpenRouter endpoints alike.

These tests drive the real :class:`OpenAILLM` and inspect the kwargs that reach
the OpenAI client. Re-implementing the routing logic inside the test would prove
nothing about the shipped code, so every assertion below runs against
``openevolve.llm.openai`` itself.
"""

import asyncio
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, patch

import pytest

from openevolve.config import LLMModelConfig
from openevolve.llm.openai import OpenAILLM

# Endpoints that must all behave identically for a given model name. The
# regional/Azure/proxy entries are the ones the original fix was about.
ENDPOINTS = [
    "https://api.openai.com/v1",
    "https://eu.api.openai.com/v1",
    "https://apac.api.openai.com/v1",
    "https://my-resource.openai.azure.com/",
    "http://localhost:8000/v1",  # OptiLLM proxy
    "https://openrouter.ai/api/v1",
]

# (model name, is_reasoning_model)
#
# Detection is a case-insensitive *prefix* match against
# ``OPENAI_REASONING_MODEL_PREFIXES`` inside ``openai.py``, so these cases lock
# in the set of model families the project actually routes.
MODEL_CASES = [
    ("o1", True),
    ("o1-mini", True),
    ("o1-preview", True),
    ("o3-mini", True),
    ("o3-pro", True),
    ("o4-mini", True),
    ("gpt-5", True),
    ("gpt-5-mini", True),
    ("gpt-5-nano", True),
    ("gpt-oss-120b", True),
    ("gpt-oss-20b", True),
    ("O1-MINI", True),  # matching is case-insensitive
    ("gpt-4", False),
    ("gpt-4o", False),
    ("gpt-3.5-turbo", False),
    ("claude-sonnet-4", False),
    ("gemini-2.0-flash", False),
    ("Qwen/Qwen2.5-7B-Instruct", False),
]


def _fake_completion(content: str = "ok") -> MagicMock:
    """Build a stand-in for an OpenAI ChatCompletion response."""
    completion = MagicMock()
    completion.choices[0].message.content = content
    completion.usage.prompt_tokens = 3
    completion.usage.completion_tokens = 5
    completion.usage.total_tokens = 8
    return completion


@pytest.fixture
def llm_factory() -> Any:
    """
    Yield ``(build, captured)`` where ``build`` constructs a real ``OpenAILLM``
    against a mocked OpenAI client and ``captured`` holds the kwargs of the most
    recent ``chat.completions.create`` call.
    """
    captured: Dict[str, Any] = {}

    def fake_create(**params: Any) -> MagicMock:
        captured.clear()
        captured.update(params)
        return _fake_completion()

    with patch("openevolve.llm.openai.openai.OpenAI") as openai_cls:
        openai_cls.return_value.chat.completions.create.side_effect = fake_create

        def build(name: str, **cfg: Any) -> OpenAILLM:
            cfg.setdefault("api_key", "test-key")
            cfg.setdefault("api_base", "https://api.openai.com/v1")
            return OpenAILLM(LLMModelConfig(name=name, **cfg))

        yield build, captured


def _generate(llm: OpenAILLM, **kwargs: Any) -> str:
    return asyncio.run(llm.generate("hello", **kwargs))


@pytest.mark.parametrize("model,is_reasoning", MODEL_CASES)
def test_parameter_routing_by_model_name(llm_factory: Any, model: str, is_reasoning: bool) -> None:
    """Reasoning models get max_completion_tokens; every other model gets max_tokens."""
    build, captured = llm_factory
    llm = build(model, max_tokens=4096, temperature=0.7)

    _generate(llm)

    if is_reasoning:
        assert captured["max_completion_tokens"] == 4096
        assert "max_tokens" not in captured
        # Reasoning models reject these outright.
        assert "temperature" not in captured
        assert "top_p" not in captured
    else:
        assert captured["max_tokens"] == 4096
        assert "max_completion_tokens" not in captured
        assert captured["temperature"] == 0.7


@pytest.mark.parametrize("api_base", ENDPOINTS)
def test_routing_is_independent_of_endpoint(llm_factory: Any, api_base: str) -> None:
    """
    Regression guard for the original bug: a reasoning model behind a regional,
    Azure, or proxy endpoint still needs max_completion_tokens.
    """
    build, captured = llm_factory

    reasoning_llm = build("o1-mini", api_base=api_base, max_tokens=100)
    _generate(reasoning_llm)
    assert captured["max_completion_tokens"] == 100
    assert "max_tokens" not in captured

    standard_llm = build("gpt-4", api_base=api_base, max_tokens=100)
    _generate(standard_llm)
    assert captured["max_tokens"] == 100
    assert "max_completion_tokens" not in captured


def test_case_insensitive_model_matching(llm_factory: Any) -> None:
    """Model names are lower-cased before matching, in both directions."""
    build, captured = llm_factory

    _generate(build("O3-MINI", max_tokens=64))
    assert captured["max_completion_tokens"] == 64

    _generate(build("GPT-4", max_tokens=64))
    assert captured["max_tokens"] == 64


def test_reasoning_effort_is_forwarded(llm_factory: Any) -> None:
    """A configured reasoning_effort reaches the request for both model families."""
    build, captured = llm_factory

    _generate(build("o3-mini", max_tokens=64, reasoning_effort="high"))
    assert captured["reasoning_effort"] == "high"

    # reasoning_effort is also supported by open-source reasoning models served
    # over OpenAI-compatible endpoints.
    _generate(build("Qwen/Qwen3-8B", max_tokens=64, reasoning_effort="low"))
    assert captured["reasoning_effort"] == "low"


def test_no_reasoning_effort_when_unset(llm_factory: Any) -> None:
    build, captured = llm_factory

    _generate(build("o3-mini", max_tokens=64))
    assert "reasoning_effort" not in captured


def test_call_kwargs_override_config(llm_factory: Any) -> None:
    """Per-call kwargs win over the values baked into the model config."""
    build, captured = llm_factory
    llm = build("gpt-4", max_tokens=1024, temperature=0.7, top_p=0.9)

    _generate(llm, max_tokens=32, temperature=0.1, top_p=0.5)

    assert captured["max_tokens"] == 32
    assert captured["temperature"] == 0.1
    assert captured["top_p"] == 0.5


def test_top_p_omitted_when_none(llm_factory: Any) -> None:
    build, captured = llm_factory

    _generate(build("gpt-4", max_tokens=64, top_p=None))
    assert "top_p" not in captured


def test_verbosity_only_for_reasoning_models(llm_factory: Any) -> None:
    """`verbosity` is a reasoning-model parameter and must not leak to others."""
    build, captured = llm_factory

    _generate(build("gpt-5", max_tokens=64), verbosity="low")
    assert captured["verbosity"] == "low"

    _generate(build("gpt-4", max_tokens=64), verbosity="low")
    assert "verbosity" not in captured


def test_default_model_config_does_not_crash_on_generate(llm_factory: Any) -> None:
    """
    Regression test: ``LLMModelConfig`` leaves ``retries``/``retry_delay`` as
    ``None`` (only ``LLMConfig`` fills them in, and only for models attached to
    it). A directly constructed model config used to reach ``range(None + 1)``
    and die with ``TypeError: unsupported operand type(s) for +`` on the very
    first ``generate()`` — even though ``__init__`` already guarded the same
    value when building the OpenAI client.
    """
    build, _ = llm_factory
    llm = build("gpt-4", max_tokens=64)  # deliberately no retries/retry_delay

    assert llm.retries == 0
    assert llm.retry_delay == 0
    assert _generate(llm) == "ok"


def test_retries_are_honoured(llm_factory: Any) -> None:
    """A transient failure is retried and the later success is returned."""
    build, _ = llm_factory
    llm = build("gpt-4", max_tokens=64, retries=2, retry_delay=0)

    attempts = {"n": 0}

    def flaky(**params: Any) -> MagicMock:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return _fake_completion("recovered")

    llm.client.chat.completions.create.side_effect = flaky

    assert _generate(llm) == "recovered"
    assert attempts["n"] == 3


def test_exhausted_retries_raise_the_last_error(llm_factory: Any) -> None:
    build, _ = llm_factory
    llm = build("gpt-4", max_tokens=64, retries=1, retry_delay=0)

    attempts = {"n": 0}

    def always_fails(**params: Any) -> MagicMock:
        attempts["n"] += 1
        raise RuntimeError(f"boom {attempts['n']}")

    llm.client.chat.completions.create.side_effect = always_fails

    with pytest.raises(RuntimeError, match="boom 2"):
        _generate(llm)
    assert attempts["n"] == 2  # initial attempt + one retry


def test_missing_system_message_is_sent_as_empty_string(llm_factory: Any) -> None:
    """
    A bare model config has no system message. The request must still carry a
    string for that role rather than JSON ``null``, which the API rejects.
    """
    build, captured = llm_factory
    llm = build("gpt-4", max_tokens=64)

    _generate(llm)

    assert captured["messages"][0] == {"role": "system", "content": ""}


def test_system_message_and_messages_are_forwarded(llm_factory: Any) -> None:
    """generate_with_context prepends the system message, then the conversation."""
    build, captured = llm_factory
    llm = build("gpt-4", max_tokens=64, system_message="be terse")

    asyncio.run(
        llm.generate_with_context(
            system_message="be terse",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert captured["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_token_usage_is_recorded(llm_factory: Any) -> None:
    """Usage from the response is surfaced on ``last_usage``."""
    build, _ = llm_factory
    llm = build("gpt-4", max_tokens=64)

    assert _generate(llm) == "ok"
    assert llm.last_usage is not None
    assert llm.last_usage["total_tokens"] == 8


def test_string_response_is_rejected(llm_factory: Any) -> None:
    """
    Some endpoints stream SSE even when asked not to; that must raise a clear
    error rather than being returned as a half-parsed string.
    """
    build, captured = llm_factory
    llm = build("gpt-4", max_tokens=64, retries=0)
    llm.client.chat.completions.create.side_effect = lambda **params: "data: {...}"

    with pytest.raises(ValueError, match="returned a raw string"):
        _generate(llm)


def test_client_is_not_constructed_in_manual_mode(tmp_path: Any, monkeypatch: Any) -> None:
    """Manual mode must skip API clients entirely and use the queue directory."""
    monkeypatch.setenv("OPENEVOLVE_MANUAL_QUEUE_DIR", str(tmp_path))

    with patch("openevolve.llm.openai.openai.OpenAI") as openai_cls:
        llm = OpenAILLM(
            LLMModelConfig(
                name="o1-mini",
                api_key="test-key",
                manual_mode=True,
                _manual_queue_dir=str(tmp_path),
            )
        )

    assert llm.client is None
    assert llm.manual_queue_dir == tmp_path.resolve()
    openai_cls.assert_not_called()
