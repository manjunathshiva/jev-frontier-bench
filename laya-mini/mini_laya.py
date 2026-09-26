"""Laya on a 16 GB Mac mini: load time, latency (GPU and CPU-only), peak memory, answers.

Needs only laya (pip install laya==0.3.20) and results/items.json + results/ext_items.json
(python bench.py sample; python ext_bench.py sample). The answers can be compared with the
M4 Max run in results/calls.jsonl and results/ext_calls.jsonl, which used the same items.

    python laya-mini/mini_laya.py   # writes results/mini/mini_laya.json and mini_answers.jsonl
"""
from __future__ import annotations

import json
import platform
import resource
import statistics as st
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "results"
RES = HERE / "mini"
RES.mkdir(exist_ok=True)


def items():
    out = json.loads((HERE / "items.json").read_text())["items"]
    for it in json.loads((HERE / "ext_items.json").read_text())["items"]:
        out.append(dict(it, options={k: (v or k) for k, v in it["options"].items()}))
    return out


def question(it):
    if it["type"] == "noul":
        return {"type": "noul", "instructions": it["instructions"]}
    if it["type"] == "choice":
        return {"type": "choice", "instructions": it["instructions"], "criteria": it["options"]}
    return {"type": "score", "instructions": it["instructions"], "criteria": list(it["options"].values())}


def answer(it, a):
    if a["type"] == "noul":
        p = {"yes": a["noul"], "no": 1 - a["noul"]}
    elif a["type"] == "choice":
        p = a["probabilities"]
    else:
        keys = list(it["options"])
        p = {keys[int(k)]: v for k, v in a["probabilities"].items()}
    return max(p, key=p.get), p


def thermal():
    try:
        return int(subprocess.check_output(
            ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], text=True).split()[-1])
    except Exception:
        return None


def run(device: str, its, log=None):
    from laya import Router
    t0 = time.perf_counter()
    r = Router(max_loaded=1, device=device)
    r.predict({"message": "warm up"}, {"q": {"type": "noul", "instructions": "Is this a test?"}},
              model="english")
    load_s = time.perf_counter() - t0
    walls, correct = [], 0
    for it in its:
        t = time.perf_counter()
        res = r.predict(it["state"], {"q": question(it)}, model="english")
        walls.append(time.perf_counter() - t)
        pick, p = answer(it, res["answers"]["q"])
        correct += pick == it["gold"]
        if log is not None:
            log.write(json.dumps({"item": it["id"], "task": it["task"], "answer": pick, "probs": p,
                                  "wall_s": walls[-1]}) + "\n")
    walls.sort()
    return {"device": device, "load_s": round(load_s, 1), "n": len(its),
            "accuracy": round(correct / len(its), 3),
            "p50_ms": round(1000 * st.median(walls), 1),
            "p95_ms": round(1000 * walls[int(0.95 * len(walls))], 1),
            "thermal_after": thermal()}


def main():
    import torch
    import laya
    its = items()
    info = {"chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "ram_gb": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) // 2**30,
            "macos": platform.mac_ver()[0], "laya": getattr(laya, "__version__", "?"),
            "torch": torch.__version__, "mps": torch.backends.mps.is_available()}
    print(json.dumps(info))
    with (RES / "mini_answers.jsonl").open("w") as log:
        info["gpu"] = run("mps", its, log)            # all 400 items on the GPU
    print(json.dumps(info["gpu"]))
    info["cpu"] = run("cpu", its[::4])                # every 4th item on CPU only
    print(json.dumps(info["cpu"]))
    # ru_maxrss is bytes on macOS
    info["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    (RES / "mini_laya.json").write_text(json.dumps(info, indent=1))
    print(f"peak RSS {info['peak_rss_gib']} GiB -> {RES / 'mini_laya.json'}")


if __name__ == "__main__":
    main()
