"""Run conditions A/B/C over the scenario benchmark. Resumable: results append to a JSONL file.

Examples:
  python scripts/run_llm_experiments.py --backend hf --model Qwen/Qwen2.5-7B-Instruct
  python scripts/run_llm_experiments.py --backend anthropic --model claude-haiku-5-5
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.network import load_network
from src.scenarios import Scenario
from src.llm_conditions import LLMClient, HFClient, run_trial

DATA = ROOT / "data"
SCEN = ROOT / "experiments" / "scenarios.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "anthropic"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", default="A,B,C")
    args = ap.parse_args()

    net = load_network(DATA / "SiouxFalls_net.tntp", DATA / "SiouxFalls_trips.tntp", "SiouxFalls")
    scenarios = [Scenario(**{k: v for k, v in d.items() if k in {"sid", "text", "op", "params", "reference"}})
                 for d in json.loads(SCEN.read_text())]
    client = HFClient(args.model) if args.backend == "hf" else LLMClient(args.model)

    out = ROOT / "experiments" / f"results_{args.model.replace('/', '_')}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["sid"], r["condition"]))

    with out.open("a") as fh:
        for sc in scenarios:
            for cond in args.conditions.split(","):
                if (sc.sid, cond) in done:
                    continue
                trial = run_trial(net, sc, cond, client)
                row = dataclasses.asdict(trial)
                row["reference_tstt"] = sc.reference["scenario_tstt"]
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(sc.sid, cond, "correct" if trial.formulation_correct else "wrong",
                      trial.predicted_tstt, flush=True)


if __name__ == "__main__":
    main()
