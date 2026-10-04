"""
Claude Code CLI interface for LLMs.

Uses the Claude Code CLI (`claude -p`) as a non-interactive LLM backend,
enabling OpenEvolve to run with Anthropic's Claude models without requiring
direct API keys — authentication is handled by the CLI's OAuth session.

Usage in config.yaml:
    llm:
      provider: "claude_code"
      models:
        - name: "sonnet"
          weight: 1.0
          max_tokens: 16000
          timeout: 300

Or inject programmatically:
    from openevolve.llm.claude_code import init_claude_code_client
    for model_cfg in config.llm.models:
        model_cfg.init_client = init_claude_code_client
"""

import asyncio
import logging
import subprocess
from typing import Any, Dict, List, Optional

from openevolve.config import LLMModelConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


def _cfg_value(model_cfg: Any, attr: str, default: Any) -> Any:
    """Read a config field, falling back when it is absent *or* None.

    `getattr(cfg, attr, default)` is not enough here: every `LLMModelConfig` field
    exists but defaults to None, so `getattr` would hand back None instead of the
    fallback.
    """
    value = getattr(model_cfg, attr, None)
    return default if value is None else value


class ClaudeCodeLLM(LLMInterface):
    """LLM interface that uses the Claude Code CLI for generation.

    Requires `claude` CLI to be installed and authenticated
    (run `claude login` first).
    """

    def __init__(self, model_cfg: Optional[LLMModelConfig] = None) -> None:
        self.model = getattr(model_cfg, "name", "sonnet")
        self.system_message = getattr(model_cfg, "system_message", None)
        self.max_tokens = getattr(model_cfg, "max_tokens", 16000)
        self.timeout = getattr(model_cfg, "timeout", 300)
        self.weight = getattr(model_cfg, "weight", 1.0)
        # A directly-constructed `LLMModelConfig` leaves retries/retry_delay as
        # None (only `LLMConfig` fills them in), which would make
        # `range(retries + 1)` raise `TypeError: unsupported operand type(s)
        # for +: 'NoneType' and 'int'` on the first call — the same defect that
        # existed in `llm/openai.py`.
        self.retries = _cfg_value(model_cfg, "retries", 3)
        self.retry_delay = _cfg_value(model_cfg, "retry_delay", 5)
        self.max_budget_usd = getattr(model_cfg, "max_budget_usd", 1.0)
        self.cwd = getattr(model_cfg, "cwd", None)
        logger.info(f"Initialized ClaudeCodeLLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        sys_msg = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system_message=sys_msg,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs: Any
    ) -> str:
        user_content = "\n\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )

        cmd = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--no-session-persistence",
            "--output-format",
            "text",
        ]
        if system_message:
            cmd.extend(["--system-prompt", system_message])

        budget = kwargs.get("max_budget_usd", self.max_budget_usd)
        cmd.extend(["--max-budget-usd", str(budget)])

        # The prompt goes through stdin: as one argv string it hits Linux's 128 KiB
        # per-argument limit (MAX_ARG_STRLEN) once a few programs are in the prompt.

        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        loop = asyncio.get_event_loop()
        for attempt in range(retries + 1):
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._run_cli(cmd, timeout, user_content)),
                    timeout=timeout + 30,
                )
                return result
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Claude Code CLI timeout on attempt {attempt + 1}/{retries + 1}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Claude Code CLI error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {e}")
                    raise

        # Unreachable for retries >= 0: the loop always runs at least once and its
        # final attempt either returns or re-raises. This keeps the function total
        # for the type checker.
        raise RuntimeError("retry loop exited without returning a response")

    def _run_cli(self, cmd: list, timeout: int, prompt: Optional[str] = None) -> str:
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                if stderr:
                    logger.warning(f"Claude CLI stderr: {stderr[:500]}")
            output = result.stdout.strip()
            if not output:
                raise RuntimeError(f"Empty response from Claude CLI. stderr: {result.stderr[:500]}")
            return output
        except subprocess.TimeoutExpired:
            raise asyncio.TimeoutError("Claude CLI subprocess timed out")


def init_claude_code_client(model_cfg: Optional[LLMModelConfig]) -> LLMInterface:
    """Factory function compatible with OpenEvolve's init_client config hook."""
    return ClaudeCodeLLM(model_cfg)
