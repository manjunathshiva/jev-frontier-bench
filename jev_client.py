"""Minimal client for TypeSafe's Jev via OpenRouter's Decisions endpoint.

Jev is not a chat model: it takes a `state` (text or JSON) and a map of typed
questions, and returns one typed answer per question.

  noul    probability that a yes/no condition holds    -> {"noul": 0.93}
  choice  one option from a map of options              -> {"choice", "probabilities", "confidence"}
  score   position on an ordered rubric (>= 2 levels)   -> {"score", "legend", "probabilities", "confidence"}

Endpoint: POST https://openrouter.ai/api/alpha/decisions (alpha path, may move).
Same body as TypeSafe's own POST https://api.typesafe.ai/v1/systemone.
Docs: https://docs.typesafe.ai/api

The key is read from OPENROUTER_API_KEY or TYPESAFE_API_KEY, in the environment
or in a .env file next to this script (chmod 600), so it never appears on a command line or in a
transcript. An OpenRouter key (sk-or-...) routes to OpenRouter; any other key to
TypeSafe's API directly.

Usage:
  python jev_client.py smoke          # three-question smoke test, prints answers + latency
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"  # pinned; "~typesafe/jev-latest" follows new releases


ENV_FILE = Path(__file__).resolve().parent / ".env"


def api_key() -> str:
    """OPENROUTER_API_KEY or TYPESAFE_API_KEY, from the environment or .env."""
    names = ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY")
    for name in names:
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() in names and value.strip():
                return value.strip().strip("\"'")
    sys.exit(f"Set OPENROUTER_API_KEY or TYPESAFE_API_KEY (env or {ENV_FILE})")


def route(key: str) -> str:
    """An OpenRouter key (sk-or-...) goes to OpenRouter; anything else to TypeSafe directly."""
    return URL if key.startswith("sk-or-") else "https://api.typesafe.ai/v1/systemone"


def evaluate(state, questions: dict, model: str = DEFAULT_MODEL, retries: int = 3) -> tuple[dict, float]:
    """POST one evaluation. Returns (response JSON, wall seconds). Retries 429/529."""
    key = api_key()
    url = route(key)
    if url != URL and model.startswith(("typesafe/", "~typesafe/")):
        model = "jev-latest"  # TypeSafe's own API names models without the provider prefix
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    delay = 1.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read()), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise SystemExit(f"jev: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def smoke() -> None:
    state = "Help! My payouts have been failing for 3 days and my rent is due tomorrow."
    questions = {
        "urgent": {"type": "noul", "instructions": "Is this ticket urgent?"},
        "team": {"type": "choice", "instructions": "Which team should own this ticket?",
                 "criteria": {"payments": "Payouts, charges, refunds",
                              "accounts": "Login, profile, verification",
                              "general": "Anything else"}},
        "severity": {"type": "score", "instructions": "How severe is the customer impact?",
                     "criteria": ["No impact", "Minor inconvenience", "Blocks a task", "Financial harm"]},
    }
    resp, wall = evaluate(state, questions)
    print(json.dumps(resp, indent=2))
    print(f"wall {wall * 1000:.0f} ms (includes network)")


if __name__ == "__main__":
    if sys.argv[1:] == ["smoke"]:
        smoke()
    else:
        print(__doc__)
