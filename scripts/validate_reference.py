"""Validate the UE solver against published Sioux Falls equilibrium flows."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.network import load_network
from src.assignment import solve_user_equilibrium

DATA = Path(__file__).resolve().parents[1] / "data"

def load_reference_flows(path):
    flows = {}
    for line in path.read_text().splitlines()[1:]:
        p = line.strip().rstrip(";").split()
        if len(p) >= 3:
            flows[(int(p[0]), int(p[1]))] = float(p[2])
    return flows

def main():
    net = load_network(DATA / "SiouxFalls_net.tntp", DATA / "SiouxFalls_trips.tntp", "SiouxFalls")
    print(f"Links: {len(net.links)}  OD pairs: {len(net.demand)}  Total trips: {sum(net.demand.values()):.0f}")
    res = solve_user_equilibrium(net, max_iter=5000, tol=1e-5)
    print(f"Iterations: {res.iterations}  relative gap: {res.relative_gap:.2e}  TSTT: {res.tstt:,.1f}")

    ref = load_reference_flows(DATA / "SiouxFalls_flow.tntp")
    idx = {(l.init, l.term): i for i, l in enumerate(net.links)}
    ours = np.array([res.flows[idx[k]] for k in ref])
    theirs = np.array([ref[k] for k in ref])
    rel_err = np.abs(ours - theirs) / np.maximum(theirs, 1.0)
    print(f"Link flow vs published: mean rel err {rel_err.mean():.4%}, max rel err {rel_err.max():.4%}")
    # TSTT scale is not compared to a published figure: the convention is unverified here.

if __name__ == "__main__":
    main()
