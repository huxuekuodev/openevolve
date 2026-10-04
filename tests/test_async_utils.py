"""Behavioural tests for ``openevolve/utils/async_utils.py``.

Covers the retry/backoff helper, the timeout helpers, the concurrency-limited
gather, the executor decorator and ``TaskPool``.

Everything here is offline and deterministic: no real sleeping is required to
observe the backoff sequence (``asyncio.sleep`` is recorded, not awaited), and
timeouts are driven by coroutines that never complete rather than by wall-clock
races.
"""

import asyncio
import logging
import threading
from typing import Any, Dict, List

import pytest

import openevolve.utils.async_utils as async_utils
from openevolve.utils.async_utils import (
    TaskPool,
    gather_with_concurrency,
    retry_async,
    run_in_executor,
    run_sync_with_timeout,
    run_with_timeout,
)

DEFAULT_TIMEOUT_VALUE = {"error": 0.0, "timeout": True}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FlakyCoroutine:
    """Async callable that raises ``failures`` times before returning ``value``.

    Keeps every exception instance it raised so tests can assert that the
    *original* exception object is re-raised after the retries are exhausted.
    """

    def __init__(self, failures: int = 0, exc: type = ValueError, value: Any = "ok"):
        self.failures = failures
        self.exc = exc
        self.value = value
        self.calls = 0
        self.raised: List[BaseException] = []
        self.received: List[Dict[str, Any]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.received.append({"args": args, "kwargs": kwargs})
        if self.calls <= self.failures:
            error = self.exc(f"attempt {self.calls}")
            self.raised.append(error)
            raise error
        return self.value


class ConcurrencyProbe:
    """Records how many coroutines were in flight at the same time."""

    def __init__(self, yields: int = 3):
        self.yields = yields
        self.active = 0
        self.max_active = 0

    async def work(self, value: Any) -> Any:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            for _ in range(self.yields):
                # Yield control so other coroutines get a chance to start.
                await asyncio.sleep(0)
            return value
        finally:
            self.active -= 1


async def never_completes(*args: Any, **kwargs: Any) -> Any:
    """Coroutine that only ends when cancelled (i.e. when the timeout fires)."""
    await asyncio.Event().wait()


async def echo(a: Any, b: Any = None, *, c: Any = None) -> Any:
    return (a, b, c)


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """Capture (and skip) every ``asyncio.sleep`` performed by retry_async."""
    delays: List[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(async_utils.asyncio, "sleep", fake_sleep)
    return delays


# ---------------------------------------------------------------------------
# retry_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_async_returns_first_success_without_sleeping(recorded_sleeps):
    flaky = FlakyCoroutine(failures=0, value="first")

    result = await retry_async(flaky, retries=3, delay=0.0)

    assert result == "first"
    assert flaky.calls == 1
    assert recorded_sleeps == []


@pytest.mark.asyncio
async def test_retry_async_forwards_args_and_kwargs():
    flaky = FlakyCoroutine(failures=1)

    await retry_async(flaky, 1, "two", c=3, retries=2, delay=0.0)

    assert flaky.received == [{"args": (1, "two"), "kwargs": {"c": 3}}] * 2


@pytest.mark.asyncio
async def test_retry_async_succeeds_after_retries(recorded_sleeps):
    flaky = FlakyCoroutine(failures=2, value="recovered")

    result = await retry_async(flaky, retries=3, delay=0.0)

    assert result == "recovered"
    assert flaky.calls == 3


@pytest.mark.asyncio
async def test_retry_async_reraises_the_original_exception_when_exhausted(recorded_sleeps):
    flaky = FlakyCoroutine(failures=10)

    with pytest.raises(ValueError) as excinfo:
        await retry_async(flaky, retries=2, delay=0.0)

    # retries=2 means 3 total attempts, and the *last* error object is re-raised.
    assert flaky.calls == 3
    assert excinfo.value is flaky.raised[-1]
    assert str(excinfo.value) == "attempt 3"
    assert len(flaky.raised) == 3


@pytest.mark.asyncio
async def test_retry_async_with_zero_retries_makes_one_attempt(recorded_sleeps):
    flaky = FlakyCoroutine(failures=10)

    with pytest.raises(ValueError):
        await retry_async(flaky, retries=0, delay=0.0)

    assert flaky.calls == 1
    assert recorded_sleeps == []


@pytest.mark.asyncio
async def test_retry_async_ignores_exceptions_outside_the_configured_class(recorded_sleeps):
    flaky = FlakyCoroutine(failures=10, exc=KeyError)

    with pytest.raises(KeyError):
        await retry_async(flaky, retries=3, delay=0.0, exceptions=ValueError)

    # Not caught -> no retry at all.
    assert flaky.calls == 1
    assert recorded_sleeps == []


@pytest.mark.asyncio
async def test_retry_async_retries_every_exception_in_a_tuple(recorded_sleeps):
    class Alternating:
        def __init__(self):
            self.calls = 0

        async def __call__(self):
            self.calls += 1
            if self.calls == 1:
                raise KeyError("k")
            if self.calls == 2:
                raise ValueError("v")
            return "done"

    coro = Alternating()

    result = await retry_async(coro, retries=3, delay=0.0, exceptions=(KeyError, ValueError))

    assert result == "done"
    assert coro.calls == 3


@pytest.mark.asyncio
async def test_retry_async_catches_subclasses_of_configured_exceptions(recorded_sleeps):
    class MyError(ValueError):
        pass

    flaky = FlakyCoroutine(failures=1, exc=MyError, value="ok")

    result = await retry_async(flaky, retries=2, delay=0.0, exceptions=ValueError)

    assert result == "ok"
    assert flaky.calls == 2


@pytest.mark.asyncio
async def test_retry_async_uses_exponential_backoff_between_attempts(recorded_sleeps):
    flaky = FlakyCoroutine(failures=10)

    with pytest.raises(ValueError):
        await retry_async(flaky, retries=3, delay=0.5, backoff=3.0)

    # 3 retries -> 3 sleeps, and no sleep after the final (failing) attempt.
    assert flaky.calls == 4
    assert recorded_sleeps == pytest.approx([0.5, 1.5, 4.5])


@pytest.mark.asyncio
async def test_retry_async_backoff_of_one_keeps_the_delay_constant(recorded_sleeps):
    flaky = FlakyCoroutine(failures=2)

    await retry_async(flaky, retries=5, delay=0.25, backoff=1.0)

    assert recorded_sleeps == pytest.approx([0.25, 0.25])


@pytest.mark.asyncio
async def test_retry_async_can_return_none(recorded_sleeps):
    flaky = FlakyCoroutine(failures=0, value=None)

    assert await retry_async(flaky, retries=2, delay=0.0) is None
    assert flaky.calls == 1


# ---------------------------------------------------------------------------
# run_with_timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_with_timeout_returns_result_and_forwards_args():
    assert await run_with_timeout(echo, 1.0, 1, 2, c=3) == (1, 2, 3)


@pytest.mark.asyncio
async def test_run_with_timeout_returns_default_value_on_timeout():
    result = await run_with_timeout(never_completes, 0.01)

    assert result == DEFAULT_TIMEOUT_VALUE
    assert result is not DEFAULT_TIMEOUT_VALUE  # a fresh dict per call


@pytest.mark.asyncio
async def test_run_with_timeout_returns_custom_timeout_value():
    result = await run_with_timeout(never_completes, 0.01, timeout_error_value="gave up")

    assert result == "gave up"


@pytest.mark.asyncio
async def test_run_with_timeout_treats_none_timeout_value_as_default():
    # timeout_error_value=None is indistinguishable from "not provided".
    result = await run_with_timeout(never_completes, 0.01, timeout_error_value=None)

    assert result == DEFAULT_TIMEOUT_VALUE


@pytest.mark.asyncio
async def test_run_with_timeout_propagates_other_exceptions():
    async def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await run_with_timeout(boom, 1.0)


@pytest.mark.asyncio
async def test_run_with_timeout_logs_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="openevolve.utils.async_utils"):
        await run_with_timeout(never_completes, 0.01)

    assert "timed out after 0.01s" in caplog.text


# ---------------------------------------------------------------------------
# run_sync_with_timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sync_with_timeout_runs_off_the_event_loop_thread():
    def report_thread(value):
        return value, threading.get_ident()

    value, thread_id = await run_sync_with_timeout(report_thread, 1.0, "hello")

    assert value == "hello"
    assert thread_id != threading.get_ident()


@pytest.mark.asyncio
async def test_run_sync_with_timeout_forwards_args_and_kwargs():
    def combine(a, b, c=0):
        return a + b + c

    assert await run_sync_with_timeout(combine, 1.0, 1, 2, c=3) == 6


@pytest.mark.asyncio
async def test_run_sync_with_timeout_returns_default_value_on_timeout():
    release = threading.Event()

    def blocking(value):
        release.wait(2.0)
        return value

    try:
        result = await run_sync_with_timeout(blocking, 0.01, "too late")
        assert result == DEFAULT_TIMEOUT_VALUE
    finally:
        # Let the (already abandoned) executor thread finish promptly.
        release.set()


@pytest.mark.asyncio
async def test_run_sync_with_timeout_returns_custom_timeout_value():
    release = threading.Event()

    def blocking():
        release.wait(2.0)

    try:
        result = await run_sync_with_timeout(blocking, 0.01, timeout_error_value={"timeout": "y"})
        assert result == {"timeout": "y"}
    finally:
        release.set()


@pytest.mark.asyncio
async def test_run_sync_with_timeout_propagates_exceptions():
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        await run_sync_with_timeout(boom, 1.0)


# ---------------------------------------------------------------------------
# gather_with_concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_with_concurrency_preserves_order_and_limits_concurrency():
    probe = ConcurrencyProbe()
    tasks = [probe.work(i) for i in range(5)]

    results = await gather_with_concurrency(2, *tasks)

    assert results == [0, 1, 2, 3, 4]
    assert probe.max_active <= 2


@pytest.mark.asyncio
async def test_gather_with_concurrency_of_one_is_sequential():
    probe = ConcurrencyProbe()
    tasks = [probe.work(i) for i in range(3)]

    results = await gather_with_concurrency(1, *tasks)

    assert results == [0, 1, 2]
    assert probe.max_active == 1


@pytest.mark.asyncio
async def test_gather_with_concurrency_returns_exceptions_when_requested():
    async def boom():
        raise RuntimeError("kaboom")

    results = await gather_with_concurrency(2, echo(1), boom(), return_exceptions=True)

    assert results[0] == (1, None, None)
    assert isinstance(results[1], RuntimeError)


@pytest.mark.asyncio
async def test_gather_with_concurrency_raises_by_default():
    async def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        await gather_with_concurrency(2, boom())


@pytest.mark.asyncio
async def test_gather_with_concurrency_without_tasks():
    assert await gather_with_concurrency(3) == []


# ---------------------------------------------------------------------------
# run_in_executor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_in_executor_wraps_a_sync_function():
    @run_in_executor
    def add(a, b=0):
        """Adds numbers."""
        return a + b

    assert add.__name__ == "add"
    assert add.__doc__ == "Adds numbers."
    assert await add(1, b=2) == 3


@pytest.mark.asyncio
async def test_run_in_executor_runs_off_the_event_loop_thread():
    @run_in_executor
    def thread_id():
        return threading.get_ident()

    assert await thread_id() != threading.get_ident()


@pytest.mark.asyncio
async def test_run_in_executor_propagates_exceptions():
    @run_in_executor
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        await boom()


# ---------------------------------------------------------------------------
# TaskPool
# ---------------------------------------------------------------------------


def test_task_pool_semaphore_is_lazy_and_reused():
    pool = TaskPool(max_concurrency=4)

    assert pool._semaphore is None
    semaphore = pool.semaphore
    assert pool._semaphore is semaphore
    assert pool.semaphore is semaphore
    assert pool.max_concurrency == 4


@pytest.mark.asyncio
async def test_task_pool_run_returns_the_result_and_honours_the_limit():
    pool = TaskPool(max_concurrency=2)
    probe = ConcurrencyProbe()

    results = await asyncio.gather(*(pool.run(probe.work, i) for i in range(4)))

    assert results == [0, 1, 2, 3]
    assert probe.max_active <= 2


@pytest.mark.asyncio
async def test_task_pool_run_serializes_with_a_single_slot():
    pool = TaskPool(max_concurrency=1)
    probe = ConcurrencyProbe()

    await asyncio.gather(*(pool.run(probe.work, i) for i in range(3)))

    assert probe.max_active == 1


@pytest.mark.asyncio
async def test_task_pool_create_task_tracks_and_forgets_tasks():
    pool = TaskPool(max_concurrency=2)

    first = pool.create_task(echo, 1)
    second = pool.create_task(echo, 2)

    assert len(pool.tasks) == 2
    assert first in pool.tasks and second in pool.tasks

    await pool.wait_all()

    assert first.result() == (1, None, None)
    assert second.result() == (2, None, None)
    assert pool.tasks == []


@pytest.mark.asyncio
async def test_task_pool_wait_all_without_tasks_is_a_no_op():
    pool = TaskPool()

    assert await pool.wait_all() is None
    assert pool.tasks == []


@pytest.mark.asyncio
async def test_task_pool_cancel_all_cancels_pending_tasks():
    pool = TaskPool(max_concurrency=2)
    task = pool.create_task(never_completes)
    await asyncio.sleep(0)  # let the task actually start

    assert task.cancelled() is False
    await pool.cancel_all()

    assert task.cancelled() is True
    assert pool.tasks == []


@pytest.mark.asyncio
async def test_task_pool_cancel_all_without_tasks_is_a_no_op():
    pool = TaskPool()

    assert await pool.cancel_all() is None
