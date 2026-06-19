"""Tests for multi-provider routing (no network required)."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import openai

from ai_subtitle_translator import providers
from ai_subtitle_translator import translator as translator_mod
from ai_subtitle_translator.providers import (
    ProvidersFile,
    ProviderSpec,
    RoutingProvider,
    _is_rate_limit_error,
    _ProviderRuntime,
    _retry_after_seconds,
    load_providers_file,
)


def _spec(name, provider="copilot", model="m", rpm=0, rpd=0, concurrency=5, cooldown=60.0):
    return ProviderSpec(
        name=name, provider=provider, model=model, api_key="k", base_url=None,
        api_mode="auto", send_temperature=True,
        requests_per_minute=rpm, requests_per_day=rpd,
        concurrency=concurrency, cooldown_seconds=cooldown,
    )


def _runtime(spec):
    return _ProviderRuntime(spec=spec, backend=None, semaphore=asyncio.Semaphore(spec.concurrency))


def _rate_limit_429():
    return RuntimeError("codex HTTP 429: too many requests")


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeBackend:
    """A drop-in chat backend; can fail on the first N calls or always."""

    def __init__(self, name="b", errors=None, always_error=None, response="ok"):
        self.name = name
        self.errors = list(errors or [])
        self.always_error = always_error
        self.response = response
        self.calls = 0
        self.models: list[str] = []
        self.temps: list[float] = []

    async def chat(self, system, messages, model, temperature):
        self.calls += 1
        self.models.append(model)
        self.temps.append(temperature)
        if self.always_error is not None:
            raise self.always_error
        if self.errors:
            err = self.errors.pop(0)
            if err is not None:
                raise err
        return self.response


def _build_router(specs, backends, strategy="failover", clock=None):
    with mock.patch.object(providers, "_build_backend", side_effect=lambda s: backends[s.name]):
        return RoutingProvider(ProvidersFile(strategy=strategy, specs=specs), time_fn=clock)


class LoadProvidersTest(unittest.TestCase):
    def _write(self, data):
        d = tempfile.mkdtemp()
        p = Path(d) / "providers.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_valid_with_defaults(self):
        p = self._write({
            "strategy": "round_robin",
            "providers": [
                {"name": "a", "provider": "codex", "model": "m1",
                 "limits": {"requests_per_minute": 60, "requests_per_day": 1000,
                            "concurrency": 3, "cooldown_seconds": 30}},
                {"name": "b", "model": "m2", "api_mode": "chat"},
            ],
        })
        pf = load_providers_file(p)
        self.assertEqual(pf.strategy, "round_robin")
        a, b = pf.specs
        self.assertEqual((a.provider, a.model, a.requests_per_minute, a.concurrency), ("codex", "m1", 60, 3))
        self.assertFalse(a.send_temperature)             # codex default
        self.assertEqual((b.provider, b.model, b.api_mode), ("copilot", "m2", "chat"))
        self.assertTrue(b.send_temperature)              # copilot default
        self.assertEqual(b.requests_per_minute, 0)       # unlimited default

    def test_env_fallback_for_key_and_url(self):
        p = self._write({"providers": [{"name": "a", "model": "m"}]})
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "envkey", "OPENAI_BASE_URL": "http://env"}):
            pf = load_providers_file(p)
        self.assertEqual(pf.specs[0].api_key, "envkey")
        self.assertEqual(pf.specs[0].base_url, "http://env")

    def test_rejections(self):
        bad = [
            [],                                                             # not an object
            {"providers": []},                                             # empty
            {"providers": [{"model": "m"}]},                               # missing name
            {"providers": [{"name": "a"}]},                                # missing model
            {"providers": [{"name": "a", "model": "m"}, {"name": "a", "model": "m"}]},  # dup
            {"strategy": "nope", "providers": [{"name": "a", "model": "m"}]},           # bad strategy
            {"providers": [{"name": "a", "model": "m", "provider": "x"}]},  # bad provider
            {"providers": [{"name": "a", "model": "m", "limits": {"requests_per_minute": -1}}]},
            {"providers": [{"name": "a", "model": "m", "limits": {"concurrency": 0}}]},
            {"providers": [{"name": "a", "model": "m", "limits": {"requests_per_day": 1.5}}]},
            {"providers": [{"name": "a", "model": "m", "limits": {"cooldown_seconds": -5}}]},
        ]
        for data in bad:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    load_providers_file(self._write(data))


class RateLimitClassifyTest(unittest.TestCase):
    def test_codex_messages(self):
        self.assertTrue(_is_rate_limit_error(RuntimeError("codex HTTP 429: x")))
        self.assertTrue(_is_rate_limit_error(RuntimeError("Quota exceeded for today")))
        self.assertTrue(_is_rate_limit_error(RuntimeError("rate limit reached")))
        self.assertFalse(_is_rate_limit_error(RuntimeError("codex HTTP 500: boom")))
        self.assertFalse(_is_rate_limit_error(RuntimeError("codex stream error: boom")))

    def test_openai_errors(self):
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        rl = openai.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)
        self.assertTrue(_is_rate_limit_error(rl))
        err500 = openai.APIStatusError("server error", response=httpx.Response(500, request=req), body=None)
        self.assertFalse(_is_rate_limit_error(err500))
        self.assertFalse(_is_rate_limit_error(ValueError("bad json")))
        self.assertFalse(_is_rate_limit_error(translator_mod.ModelNotSupportedError()))

    def test_retry_after(self):
        req = httpx.Request("POST", "http://x")
        with_header = openai.RateLimitError(
            "x", response=httpx.Response(429, headers={"retry-after": "30"}, request=req), body=None
        )
        self.assertEqual(_retry_after_seconds(with_header), 30.0)
        without = openai.RateLimitError("x", response=httpx.Response(429, request=req), body=None)
        self.assertIsNone(_retry_after_seconds(without))
        self.assertIsNone(_retry_after_seconds(RuntimeError("codex HTTP 429")))


class ProviderRuntimeTest(unittest.TestCase):
    def test_rpm_gate(self):
        rt = _runtime(_spec("a", rpm=2))
        rt.record_request(1000.0)
        rt.record_request(1000.0)
        self.assertFalse(rt.is_available(1000.0))
        self.assertTrue(rt.is_available(1061.0))  # 60s window elapsed

    def test_daily_gate_is_terminal(self):
        rt = _runtime(_spec("a", rpd=2))
        rt.record_request(1000.0)
        rt.record_request(1000.0)
        self.assertFalse(rt.is_available(1000.0))
        self.assertEqual(rt.next_available_at(1000.0), float("inf"))

    def test_cooldown(self):
        rt = _runtime(_spec("a", cooldown=60.0))
        rt.start_cooldown(1000.0)
        self.assertFalse(rt.is_available(1000.0))
        self.assertTrue(rt.is_available(1060.0))
        rt.start_cooldown(1000.0, retry_after=120.0)
        self.assertFalse(rt.is_available(1100.0))


class RoutingProviderTest(unittest.TestCase):
    def test_failover_switches_on_rate_limit(self):
        a = FakeBackend("a", errors=[_rate_limit_429()])
        b = FakeBackend("b", response="ok-b")
        router = _build_router([_spec("a"), _spec("b")], {"a": a, "b": b}, clock=Clock())
        result = asyncio.run(router.chat("sys", [{"role": "user", "content": "x"}], model="ignored", temperature=0.3))
        self.assertEqual(result, "ok-b")
        self.assertEqual((a.calls, b.calls), (1, 1))

    def test_non_rate_limit_error_reraises(self):
        a = FakeBackend("a", errors=[ValueError("bad json")])
        b = FakeBackend("b")
        router = _build_router([_spec("a"), _spec("b")], {"a": a, "b": b}, clock=Clock())
        with self.assertRaises(ValueError):
            asyncio.run(router.chat("s", [{"role": "user", "content": "x"}], model="m", temperature=0.0))
        self.assertEqual(b.calls, 0)  # the second provider was never tried

    def test_round_robin_distributes(self):
        a = FakeBackend("a")
        b = FakeBackend("b")
        router = _build_router([_spec("a"), _spec("b")], {"a": a, "b": b}, strategy="round_robin", clock=Clock())

        async def run_four():
            for _ in range(4):
                await router.chat("s", [{"role": "user", "content": "x"}], model="m", temperature=0.0)

        asyncio.run(run_four())
        self.assertEqual((a.calls, b.calls), (2, 2))

    def test_all_exhausted_raises(self):
        a = FakeBackend("a", always_error=_rate_limit_429())
        b = FakeBackend("b", always_error=_rate_limit_429())
        router = _build_router([_spec("a"), _spec("b")], {"a": a, "b": b}, clock=Clock())
        with mock.patch("asyncio.sleep", mock.AsyncMock()):
            with self.assertRaises(RuntimeError):
                asyncio.run(router.chat("s", [{"role": "user", "content": "x"}], model="m", temperature=0.0))

    def test_uses_spec_model_not_passed_model(self):
        a = FakeBackend("a")
        router = _build_router([_spec("a", model="real-model")], {"a": a}, clock=Clock())
        asyncio.run(router.chat("s", [{"role": "user", "content": "x"}], model="WRONG", temperature=0.0))
        self.assertEqual(a.models, ["real-model"])

    def test_usage_report(self):
        a = FakeBackend("a")
        b = FakeBackend("b")
        router = _build_router([_spec("a"), _spec("b")], {"a": a, "b": b}, strategy="round_robin", clock=Clock())

        async def run_two():
            await router.chat("s", [{"role": "user", "content": "x"}], model="m", temperature=0.0)
            await router.chat("s", [{"role": "user", "content": "x"}], model="m", temperature=0.0)

        asyncio.run(run_two())
        self.assertEqual(router.usage_report(), {"a": 1, "b": 1})


if __name__ == "__main__":
    unittest.main()
