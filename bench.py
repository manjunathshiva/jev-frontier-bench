"""Jev vs frontier LLMs on typed decisions.

  python bench.py sample             # rebuild the item set from the public sources -> results/items.json
                                     # (checked against results/items_manifest.json)
  python bench.py run --per-task 5   # pilot: first 5 items of each task, every arm
  python bench.py run                # full run: all 50 items per task (skips items already logged)
  python bench.py spend              # total billed so far

Every model sees the same items, instructions and options. Jev gets them as native
typed questions; the LLMs get them in one fixed prompt and must return, as JSON
schema output, a probability for every option. So every arm yields a distribution
over the same options, scored the same way.

Each call is logged as one JSON line in paper/jev_bench/calls.jsonl (request
summary, parsed answer, wall time, billed cost, raw response). Metrics are
computed from that log. Runs resume: an (item, arm) pair already logged without
error is not called again. Spending across all runs stops at BUDGET_USD.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jev_client import api_key, evaluate  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
ITEMS = OUT / "items.json"            # holds dataset text: not committed (see README)
MANIFEST = OUT / "items_manifest.json"  # ids, answers, human splits, hashes of the text
LOG = OUT / "calls.jsonl"
SEED = 20260919
PER_TASK = 50
BUDGET_USD = 12.0

ARMS = {
    "jev": "typesafe/jev-1.13",
    "fable": "anthropic/claude-fable-5.1",
    "astra": "openai/gpt-6-astra",
    "kimi": "moonshotai/kimi-k3",
    "minimax": "minimax/minimax-m3",
    "deepseek": "deepseek/deepseek-v4.1-flash",
}
# One host per model, no fallback, so speed and quantization don't vary call to call.
# First-party where it supports JSON-schema output; otherwise a named host.
PROVIDERS = {
    "fable": "anthropic",
    "astra": "openai",
    "kimi": "moonshotai",      # Moonshot AI, first-party (mxfp4)
    "minimax": "together",     # MiniMax's own endpoint lacks structured outputs
    "deepseek": "deepinfra",   # fp8; DeepSeek's own endpoint lacks structured outputs
}
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
# {"enabled": True} lets each model decide whether to think (they mostly don't on short
# questions); {"effort": "low"|"medium"|"high"} makes them think. Set by --effort.
REASONING: dict = {"enabled": True}

# ---------------------------------------------------------------- items

BANKING_CSV = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
CACHE = OUT / "cache"


def _parquet(dataset: str, config: str, split: str) -> list[dict]:
    """Download (once) and read a split from Hugging Face's parquet conversion."""
    import pandas as pd
    path = CACHE / f"{dataset.replace('/', '__')}__{config}__{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}/0.parquet"
        urllib.request.urlretrieve(url, path)
    return pd.read_parquet(path).to_dict("records")


NLI = {
    "entailment": "The hypothesis is definitely true given the premise",
    "neutral": "The hypothesis might be true or false given the premise",
    "contradiction": "The hypothesis is definitely false given the premise",
}
STARS = ["1 star: very negative", "2 stars: negative", "3 stars: mixed or neutral",
         "4 stars: positive", "5 stars: very positive"]


def sample() -> None:
    rng = random.Random(SEED)
    items = []

    # PolyAI's original CSV (the HF copy needs a loading script the rows API no longer runs).
    # One label carries a trailing "?" in the source; it is stripped here. All 77 intents
    # are offered, plus "other", as TypeSafe's docs recommend and as published BANKING77
    # results use, so the numbers are comparable.
    with urllib.request.urlopen(BANKING_CSV, timeout=60) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode())))
    for r in rows:
        r["category"] = r["category"].rstrip("?")
    intents = {k: None for k in sorted({r["category"] for r in rows})}
    intents["other"] = "None of the intents above"
    for i, r in enumerate(rng.sample(rows, PER_TASK)):
        items.append({"id": f"route-{i:02d}", "task": "route", "type": "choice",
                      "state": {"message": r["text"]},
                      "instructions": "Which intent does the bank customer's `message` express?",
                      "options": intents, "gold": r["category"]})

    rows = _parquet("google/boolq", "default", "validation")
    yes = [r for r in rows if r["answer"]]
    no = [r for r in rows if not r["answer"]]
    bq = rng.sample(yes, PER_TASK // 2) + rng.sample(no, PER_TASK // 2)
    rng.shuffle(bq)
    for i, r in enumerate(bq):
        q = r["question"].strip().rstrip("?") + "?"
        items.append({"id": f"yesno-{i:02d}", "task": "yesno", "type": "noul",
                      "state": {"passage": r["passage"]},
                      "instructions": f"According to `passage`, is the answer to this question yes? {q}",
                      "options": {"yes": "Yes", "no": "No"},
                      "gold": "yes" if bool(r["answer"]) else "no"})

    stars = {s: [] for s in range(5)}
    for r in _parquet("Yelp/yelp_review_full", "yelp_review_full", "test"):
        if len(r["text"]) <= 1500:  # keeps prompt cost bounded; noted in the article
            stars[int(r["label"])].append(r)
    yl = [x for s in range(5) for x in rng.sample(stars[s], PER_TASK // 5)]
    rng.shuffle(yl)
    for i, r in enumerate(yl):
        items.append({"id": f"rate-{i:02d}", "task": "rate", "type": "score",
                      "state": {"review": r["text"]},
                      "instructions": "How many stars did the author of `review` give?",
                      "options": {str(k + 1): d for k, d in enumerate(STARS)},
                      "gold": str(int(r["label"]) + 1)})

    rows = _parquet("metaeval/chaos-mnli-ambiguity", "default", "train")
    names = {"e": "entailment", "n": "neutral", "c": "contradiction"}
    for i, r in enumerate(rng.sample(rows, PER_TASK)):
        items.append({"id": f"ambig-{i:02d}", "task": "ambig", "type": "choice",
                      "state": {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                      "instructions": "Given `premise`, is `hypothesis` true, false, or undetermined?",
                      "options": NLI, "gold": names[r["majority_label"]],
                      "human": {names[k]: int(v or 0) / 100 for k, v in r["label_counter"].items()}})

    OUT.mkdir(parents=True, exist_ok=True)
    ITEMS.write_text(json.dumps({"seed": SEED, "items": items}, indent=1))
    print(f"{len(items)} items -> {ITEMS}")
    if MANIFEST.exists():
        verify(items)


def _sha(x) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify(items: list) -> None:
    """Check regenerated items against the published manifest, item by item."""
    want = {m["id"]: m for m in json.loads(MANIFEST.read_text())["items"]}
    bad = [it["id"] for it in items
           if it["id"] not in want
           or _sha(it["state"]) != want[it["id"]]["state_sha256"]
           or _sha(it["instructions"]) != want[it["id"]]["instructions_sha256"]
           or it["gold"] != want[it["id"]]["gold"]]
    if bad or len(items) != len(want):
        raise SystemExit(f"{len(bad)} items differ from the manifest (first: {bad[:5]}); "
                         "an upstream dataset may have changed")
    print(f"all {len(items)} items match the published manifest")


# ---------------------------------------------------------------- arms

def run_jev(item: dict) -> dict:
    if item["type"] == "noul":
        q = {"type": "noul", "instructions": item["instructions"]}
    elif item["type"] == "choice":
        q = {"type": "choice", "instructions": item["instructions"], "criteria": item["options"]}
    else:
        q = {"type": "score", "instructions": item["instructions"],
             "criteria": list(item["options"].values())}
    resp, wall = evaluate(item["state"], {"q": q}, model=ARMS["jev"])
    a = resp["answers"]["q"]
    if a["type"] == "noul":
        probs = {"yes": a["noul"], "no": 1 - a["noul"]}
    elif a["type"] == "choice":
        probs = a["probabilities"]
    else:  # score: levels are 0-based indices into the rubric
        keys = list(item["options"])
        probs = {keys[int(k)]: v for k, v in a["probabilities"].items()}
    return {"probs": probs, "wall_s": wall, "cost": resp.get("usage", {}).get("cost"),
            "usage": resp.get("usage"), "provider": resp.get("provider"), "raw": resp}


PROMPT = """Input:
{state}

Question: {instructions}

Options:
{options}

{ask}"""
FULL_ASK = ("Give a probability for every option, reflecting how likely each is to be the correct "
            "answer. The probabilities must sum to 1.")
COMPACT_ASK = ("Pick the option most likely to be correct, and give your confidence: the probability, "
               "from 0 to 1, that your pick is the correct answer.")
MAX_FULL_DIST = 10


def run_llm(arm: str, item: dict) -> dict:
    state = item["state"] if isinstance(item["state"], str) else json.dumps(item["state"], indent=1)
    options = "\n".join(f"- {k}: {v}" if v else f"- {k}" for k, v in item["options"].items())
    compact = len(item["options"]) > MAX_FULL_DIST
    if compact:
        # A probability per option would mean ~80 numbers per answer: Anthropic refuses the
        # grammar, and the output tokens dominate cost. Ask for the answer and one confidence.
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["answer", "confidence"],
                  "properties": {"answer": {"type": "string", "enum": list(item["options"])},
                                 "confidence": {"type": "number"}}}
        ask = COMPACT_ASK
    else:
        schema = {"type": "object", "additionalProperties": False, "required": ["probabilities"],
                  "properties": {"probabilities": {
                      "type": "object", "additionalProperties": False,
                      "required": list(item["options"]),
                      "properties": {k: {"type": "number"} for k in item["options"]}}}}
        ask = FULL_ASK
    body = {
        "model": ARMS[arm],
        "messages": [
            {"role": "system", "content": "You make one decision for a software system. Answer only with the requested JSON."},
            {"role": "user", "content": PROMPT.format(state=state, instructions=item["instructions"], options=options, ask=ask)},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "decision", "strict": True, "schema": schema}},
        "reasoning": REASONING,
        "max_tokens": 16000,
        "usage": {"include": True},
        "provider": {"order": [PROVIDERS[arm]], "allow_fallbacks": False},
    }
    req = urllib.request.Request(CHAT_URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
    delay = 2.0
    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            wall = time.perf_counter() - t0
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 529) and attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            # Dropped connections are retried; the wall time of the successful try is what's recorded.
            if attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    msg = resp["choices"][0]["message"]
    text = msg.get("content") or ""
    parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])
    probs = {parsed["answer"]: parsed["confidence"]} if compact else parsed["probabilities"]
    usage = resp.get("usage", {})
    return {"probs": probs, "wall_s": wall, "cost": usage.get("cost"), "usage": usage,
            "provider": resp.get("provider"), "finish": resp["choices"][0].get("finish_reason"),
            "raw": resp}


# ---------------------------------------------------------------- run

_lock = threading.Lock()


def spent() -> float:
    """Billed cost of every call in every log (runs, aborted runs, probes), for the budget."""
    total = 0.0
    for path in sorted(OUT.rglob("*.jsonl")):
        if path.exists():
            total += sum((json.loads(l).get("cost") or 0.0) for l in path.read_text().splitlines() if l.strip())
    return total


def done_pairs() -> set:
    if not LOG.exists():
        return set()
    out = set()
    for l in LOG.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if not r.get("error"):
                out.add((r["item"], r["arm"]))
    return out


def call(arm: str, item: dict) -> dict:
    rec = {"item": item["id"], "task": item["task"], "arm": arm, "model": ARMS[arm],
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        rec.update(run_jev(item) if arm == "jev" else run_llm(arm, item))
        p = rec["probs"]
        total = sum(p.values())
        rec["answer"] = max(p, key=p.get)
        rec["prob_sum"] = total
    except Exception as e:  # logged and counted as a failure for that arm
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
    with _lock, LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def run(per_task: int | None, arms: list[str]) -> None:
    items = json.loads(ITEMS.read_text())["items"]
    if per_task:
        seen = {}
        items = [it for it in items if seen.setdefault(it["task"], []).append(it) or
                 len(seen[it["task"]]) <= per_task]
    done = done_pairs()
    todo = [it for it in items if any((it["id"], a) not in done for a in arms)]
    print(f"{len(todo)} items to run x {len(arms)} arms; spent so far ${spent():.4f} of ${BUDGET_USD}")
    for n, it in enumerate(todo, 1):
        if spent() >= BUDGET_USD:
            print(f"budget reached (${spent():.2f}); stopping")
            return
        # The arms run concurrently for one item, so each arm's latency is its own.
        with cf.ThreadPoolExecutor(len(arms)) as ex:
            recs = list(ex.map(lambda a: call(a, it), [a for a in arms if (it["id"], a) not in done]))
        line = " ".join(f"{r['arm']}={'ERR' if r.get('error') else r['answer']}" for r in recs)
        print(f"[{n}/{len(todo)}] {it['id']} gold={it['gold']} | {line} | ${spent():.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "run", "spend"])
    ap.add_argument("--per-task", type=int)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--effort", choices=["low", "medium", "high"],
                    help="force reasoning at this effort (default: model decides)")
    ap.add_argument("--log", help="write to paper/jev_bench/<name>.jsonl instead of calls.jsonl")
    a = ap.parse_args()
    if a.effort:
        REASONING = {"effort": a.effort}
    if a.log:
        LOG = OUT / f"{a.log}.jsonl"
    if a.cmd == "sample":
        sample()
    elif a.cmd == "run":
        run(a.per_task, a.arms.split(","))
    else:
        print(f"${spent():.4f}")
