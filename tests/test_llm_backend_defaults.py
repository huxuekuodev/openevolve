"""
Regression tests for retry defaults in every LLM backend.

Every ``LLMModelConfig`` field exists but defaults to ``None``; only ``LLMConfig``
fills them in (via ``update_model_params``) for the models attached to it. A
``LLMModelConfig`` constructed directly -- a documented, supported usage -- keeps
``retries=None``.

``getattr(cfg, "retries", 3)`` then returns ``None`` rather than the fallback,
because the attribute *is* present. The retry loops in the backends do
``for attempt in range(retries + 1)``, so the first ``generate()`` died with
``TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'`` -- while
``__init__`` in the same class already guarded the identical value when building
the API client, which is what gave the bug away.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from openevolve.config import LLMConfig, LLMModelConfig

RETRY_DEFAULTS = [
    ("openevolve.llm.openai.OpenAILLM", 0, 0),
    ("openevolve.llm.claude_code.ClaudeCodeLLM", 3, 5),
    ("openevolve.llm.copilot_cli.CopilotCLILLM", 3, 5),
]


def _load(path):
    module_name, class_name = path.rsplit(".", 1)
    import importlib

    return getattr(importlib.import_module(module_name), class_name)


def _construct(path):
    """Build a backend from a bare LLMModelConfig (no LLMConfig in the loop)."""
    cls = _load(path)
    cfg = LLMModelConfig(name="test-model", api_key="test-key")
    with patch("openevolve.llm.openai.openai.OpenAI"):
        return cls(cfg)


@pytest.mark.parametrize("path,expected_retries,expected_delay", RETRY_DEFAULTS)
def test_direct_model_config_gets_concrete_retry_values(
    path: str, expected_retries: int, expected_delay: int
) -> None:
    """
    A bare model config must never leave retries/retry_delay as None, because the
    retry loop does arithmetic on them.
    """
    llm = _construct(path)

    assert llm.retries == expected_retries
    assert isinstance(llm.retries, int)
    assert llm.retry_delay == expected_delay
    assert isinstance(llm.retry_delay, int)


@pytest.mark.parametrize("path,_,__", RETRY_DEFAULTS)
def test_retry_loop_arithmetic_does_not_raise(path: str, _: int, __: int) -> None:
    """
    The exact expression the retry loops use must work for the constructed value.
    This is the assertion that failed before the fix.
    """
    llm = _construct(path)

    assert list(range(llm.retries + 1)) == list(range(llm.retries + 1))
    assert llm.retries + 1 >= 1


def test_explicit_retries_are_not_overridden() -> None:
    """A caller-supplied value must survive; only None is replaced."""
    from openevolve.llm.claude_code import ClaudeCodeLLM
    from openevolve.llm.copilot_cli import CopilotCLILLM
    from openevolve.llm.openai import OpenAILLM

    cfg = LLMModelConfig(name="m", api_key="k", retries=7, retry_delay=2)
    with patch("openevolve.llm.openai.openai.OpenAI"):
        for cls in (OpenAILLM, ClaudeCodeLLM, CopilotCLILLM):
            llm = cls(cfg)
            assert llm.retries == 7
            assert llm.retry_delay == 2


def test_zero_retries_is_preserved() -> None:
    """0 is a meaningful value (fail fast) and must not be treated as unset."""
    from openevolve.llm.claude_code import ClaudeCodeLLM

    llm = ClaudeCodeLLM(LLMModelConfig(name="m", retries=0, retry_delay=0))

    assert llm.retries == 0
    assert llm.retry_delay == 0
    assert list(range(llm.retries + 1)) == [0]


def test_llm_config_path_still_propagates_shared_values() -> None:
    """
    The normal YAML path keeps working: `LLMConfig.__post_init__` pushes the
    llm-level defaults into every attached model, which is why a real run never
    saw the None issue.
    """
    config = LLMConfig(models=[LLMModelConfig(name="a"), LLMModelConfig(name="b")])

    assert config.retries == 3
    assert config.retry_delay == 5
    assert [m.retries for m in config.models] == [3, 3]
    assert [m.retry_delay for m in config.models] == [5, 5]


def test_update_model_params_fills_only_unset_fields() -> None:
    """An explicit per-model value wins; only `None` fields take the shared value."""
    config = LLMConfig()
    explicit = LLMModelConfig(name="a", retries=9)
    unset = LLMModelConfig(name="b")
    config.models = [explicit, unset]

    config.update_model_params({"retries": 4})

    assert explicit.retries == 9  # explicit per-model value is preserved
    assert unset.retries == 4  # None is filled from the shared config


def test_openai_generate_works_with_a_bare_model_config() -> None:
    """
    End-to-end guard for the original symptom: build a real OpenAILLM from a bare
    config and make sure generate() reaches the (mocked) API instead of blowing up
    while computing the retry range.
    """
    from openevolve.llm.openai import OpenAILLM

    captured = {}

    def fake_create(**params):
        captured.update(params)
        completion = MagicMock()
        completion.choices[0].message.content = "ok"
        completion.usage = None
        return completion

    with patch("openevolve.llm.openai.openai.OpenAI") as openai_cls:
        openai_cls.return_value.chat.completions.create.side_effect = fake_create
        llm = OpenAILLM(LLMModelConfig(name="o1-mini", api_key="k", max_tokens=64))

        assert asyncio.run(llm.generate("hi")) == "ok"

    assert captured["max_completion_tokens"] == 64
