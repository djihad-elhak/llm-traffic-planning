"""Generate the scenario benchmark with reference answers, resumable.

Each scenario's reference is cached to experiments/scenario_cache/ as soon as it is
computed, so a run that stops partway can be resumed by running the script again.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.network import load_network
from src.assignment import solve_user_equilibrium
from src.scenarios import build_scenarios, scenario_to_dict, reference_metrics

DATA = ROOT / "data"
OUT = ROOT / "experiments" / "scenarios.json"
CACHE = ROOT / "experiments" / "scenario_cache"


def main(max_new: int = 8):
    CACHE.mkdir(parents=True, exist_ok=True)
    base = load_network(DATA / "SiouxFalls_net.tntp", DATA / "SiouxFalls_trips.tntp", "SiouxFalls")
    scenarios = build_scenarios(base, n=40, seed=7, compute_reference=False)

    base_path = CACHE / "baseline.json"
    if base_path.exists():
        base_tstt = json.loads(base_path.read_text())["tstt"]
    else:
        base_tstt = solve_user_equilibrium(base, max_iter=5000, tol=1e-5).tstt
        base_path.write_text(json.dumps({"tstt": base_tstt}))

    new = 0
    for sc in scenarios:
        f = CACHE / f"{sc.sid}.json"
        if f.exists():
            sc.reference = json.loads(f.read_text())
            continue
        if new >= max_new:
            continue
        sc.reference = reference_metrics(base, sc, base_tstt)
        f.write_text(json.dumps(sc.reference))
        new += 1
        print(f"computed {sc.sid} {sc.reference['tstt_change_pct']:+.2f}%", flush=True)

    done = [s for s in scenarios if s.reference]
    print(f"{len(done)}/{len(scenarios)} scenarios have references", flush=True)
    if len(done) == len(scenarios):
        OUT.write_text(json.dumps([scenario_to_dict(s) for s in scenarios], indent=2))
        print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main(max_new=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
