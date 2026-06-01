"""Shared LLM-call harness for all four 2024 SIOP tasks.

The contract is intentionally minimal so that the same harness handles
classification (empathy, fairness), regression (clarity), and generation
(interview). Adapters in adapters.py build the messages and parse the
responses; this module only knows how to talk to the API.

Design notes:

- Caching is by (model, messages, response_format) so re-running notebook
  cells doesn't burn money. The cache is on-disk under .harness_cache/
  next to the run, and is keyed by a hash of the request payload.

- Retries cover network errors and rate limits, but NOT bad responses.
  If the model returns "yes" when the parser wants 0/1, that's the
  adapter's problem to handle, not the harness.

- Self-consistency is a wrapper around N independent calls with the
  reducer specified by the caller (mode for classification, mean for
  regression, embedding-mean-pool for generation). It's off by default
  because the PAID Team didn't use it; Akben did, and that's what got
  them the empathy win — see KNOWN_LANDMINES.md Landmine 3.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError
except ImportError:
    # Allow this module to be imported even without openai installed,
    # so that --selftest and docs-only flows work.
    OpenAI = None  # type: ignore
    APIError = RateLimitError = APIConnectionError = Exception  # type: ignore


CACHE_DIR = Path(os.environ.get("HARNESS_CACHE_DIR", ".harness_cache"))
DEFAULT_MODEL = "gpt-4o-2024-08-06"


@dataclass
class CallSpec:
    """A single OpenAI chat-completion request, fully specified.

    The intent is that two CallSpecs with identical fields produce
    identical (cached) responses. The cache key is a hash of all fields
    that affect the model's output — model, messages, temperature,
    response_format, seed.
    """

    messages: list[dict]
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    response_format: dict | None = None
    seed: int | None = 1234
    max_tokens: int = 1024

    def cache_key(self) -> str:
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "response_format": self.response_format,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class Harness:
    """Shared LLM-call harness.

    Usage:
        h = Harness()
        text = h.call(CallSpec(messages=[...]))
        # or for self-consistency:
        winner = h.call_consistent(CallSpec(...), n=5, reduce=mode_reducer)
    """

    api_key: str | None = None
    cache_enabled: bool = True
    max_retries: int = 4
    base_backoff: float = 1.5  # seconds; doubles each retry
    log_fn: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr)
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if OpenAI is None:
            # Stay importable for selftest / docs builds.
            return
        self._client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- public API ---

    def call(self, spec: CallSpec) -> str:
        """Single chat-completion call. Returns the assistant text."""
        if self.cache_enabled:
            cached = self._cache_get(spec)
            if cached is not None:
                return cached
        text = self._call_with_retries(spec)
        if self.cache_enabled:
            self._cache_put(spec, text)
        return text

    def call_consistent(
        self,
        spec: CallSpec,
        n: int = 5,
        reduce: Callable[[list[str]], str] = None,  # type: ignore
    ) -> str:
        """N independent calls, reduced by `reduce`. Defaults to mode.

        This is the Akben-style self-consistency wrapper. Each of the N
        calls uses a distinct seed (spec.seed + i), so the harness cache
        treats them as separate requests. With temperature=0 and the same
        seed, OpenAI's chat completions are nominally deterministic, so
        for self-consistency to be meaningful we usually want
        temperature > 0 — Akben used 0.7 or 1.0 in their reproduction.
        """
        if reduce is None:
            reduce = mode_reducer
        results: list[str] = []
        for i in range(n):
            child = CallSpec(
                messages=spec.messages,
                model=spec.model,
                temperature=spec.temperature,
                response_format=spec.response_format,
                seed=None if spec.seed is None else spec.seed + i,
                max_tokens=spec.max_tokens,
            )
            results.append(self.call(child))
        return reduce(results)

    # --- internals ---

    def _call_with_retries(self, spec: CallSpec) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=spec.model,
                    messages=spec.messages,
                    temperature=spec.temperature,
                    response_format=spec.response_format,
                    seed=spec.seed,
                    max_tokens=spec.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APIConnectionError) as e:
                last_err = e
                wait = self.base_backoff * (2 ** attempt)
                self.log_fn(f"[harness] transient error, sleeping {wait:.1f}s: {e}")
                time.sleep(wait)
            except APIError as e:
                # Non-transient errors don't get retried, but we log and re-raise.
                self.log_fn(f"[harness] non-transient API error: {e}")
                raise
        raise RuntimeError(f"call failed after {self.max_retries} retries") from last_err

    def _cache_path(self, spec: CallSpec) -> Path:
        return CACHE_DIR / f"{spec.cache_key()}.txt"

    def _cache_get(self, spec: CallSpec) -> str | None:
        p = self._cache_path(spec)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def _cache_put(self, spec: CallSpec, text: str) -> None:
        self._cache_path(spec).write_text(text, encoding="utf-8")


# --- reducers for self-consistency ---


def mode_reducer(xs: list[str]) -> str:
    """Most common string. Ties broken by first-seen order."""
    if not xs:
        return ""
    counts = Counter(xs)
    return counts.most_common(1)[0][0]


def mean_reducer(xs: list[str]) -> str:
    """Parse each as float, return mean as string. For clarity task."""
    vals: list[float] = []
    for x in xs:
        try:
            vals.append(float(x.strip()))
        except (ValueError, AttributeError):
            continue
    if not vals:
        return ""
    return f"{sum(vals) / len(vals):.6f}"


def first_reducer(xs: list[str]) -> str:
    """No-op reducer; useful when self-consistency is disabled but the
    code path is the same. Returns the first item.
    """
    return xs[0] if xs else ""


# --- self-test ---


def _selftest() -> int:
    """Light sanity check; no API calls. Exits 0 on success."""
    spec_a = CallSpec(messages=[{"role": "user", "content": "hello"}])
    spec_b = CallSpec(messages=[{"role": "user", "content": "hello"}])
    spec_c = CallSpec(messages=[{"role": "user", "content": "world"}])
    assert spec_a.cache_key() == spec_b.cache_key(), "identical specs must share key"
    assert spec_a.cache_key() != spec_c.cache_key(), "differing messages must differ"

    assert mode_reducer(["yes", "yes", "no"]) == "yes"
    assert mode_reducer([]) == ""
    assert mean_reducer(["3.0", "4.0", "5.0"]) == "4.000000"
    assert mean_reducer(["bad", "also bad"]) == ""
    assert first_reducer(["a", "b"]) == "a"
    print("harness selftest OK")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Use --selftest to run sanity checks without hitting the API.")
