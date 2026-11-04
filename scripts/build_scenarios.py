"""Generate the scenario benchmark with reference answers and save as JSON."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.network import load_network
from src.scenarios import build_scenarios, scenario_to_dict

DATA = ROOT / "data"
OUT = ROOT / "experiments" / "scenarios.json"


def main():
    base = load_network(DATA / "SiouxFalls_net.tntp", DATA / "SiouxFalls_trips.tntp", "SiouxFalls")
    scenarios = build_scenarios(base, n=40, seed=7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps([scenario_to_dict(s) for s in scenarios], indent=2))
    print(f"Wrote {len(scenarios)} scenarios to {OUT}")
    for s in scenarios[:5]:
        print(s.sid, s.op, f"{s.reference['tstt_change_pct']:+.2f}%", "|", s.text)


if __name__ == "__main__":
    main()
