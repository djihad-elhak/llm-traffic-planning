"""Figures for the technical paper, generated from the repository's own data."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.network import load_network
from src.assignment import solve_user_equilibrium

DATA = ROOT / "data"
OUT = ROOT / "paper"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: validation against published equilibrium flows
net = load_network(DATA / "SiouxFalls_net.tntp", DATA / "SiouxFalls_trips.tntp", "SiouxFalls")
res = solve_user_equilibrium(net, max_iter=5000, tol=1e-5)
ref = {}
for line in (DATA / "SiouxFalls_flow.tntp").read_text().splitlines()[1:]:
    p = line.strip().rstrip(";").split()
    if len(p) >= 3:
        ref[(int(p[0]), int(p[1]))] = float(p[2])
idx = {(l.init, l.term): i for i, l in enumerate(net.links)}
ours = np.array([res.flows[idx[k]] for k in ref])
theirs = np.array(list(ref.values()))
rel = np.abs(ours - theirs) / np.maximum(theirs, 1)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.scatter(theirs, ours, s=12, color="#1f4e79", alpha=0.8)
lim = [0, max(theirs.max(), ours.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
ax.set_xlabel("Published equilibrium link flow (veh)")
ax.set_ylabel("Computed equilibrium link flow (veh)")
ax.legend(frameon=False, loc="upper left")
ax.set_title(f"Mean rel. error {rel.mean():.3%}, max {rel.max():.3%}", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_validation.pdf"); plt.close(fig)
stats = {"mean_rel_err": float(rel.mean()), "max_rel_err": float(rel.max()),
         "relative_gap": float(res.relative_gap), "iterations": int(res.iterations),
         "n_links": int(len(ref))}
(OUT / "validation_stats.json").write_text(json.dumps(stats, indent=2))

# Figure 2: benchmark TSTT change by intervention type
sc = json.loads((ROOT / "experiments" / "scenarios.json").read_text())
groups = {"close_link": "Close link", "capacity": "Capacity", "toll": "Toll", "demand": "Demand"}
data = [[s["reference"]["tstt_change_pct"] for s in sc if s["op"] == k] for k in groups]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.boxplot(data, tick_labels=list(groups.values()), showfliers=True)
for i, d in enumerate(data, 1):
    ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(d)), d, s=10, alpha=0.6, color="#1f4e79")
ax.axhline(0, color="grey", lw=0.6)
ax.set_ylabel("Change in total system travel time (%)")
fig.tight_layout(); fig.savefig(OUT / "fig_benchmark.pdf"); plt.close(fig)
print("figures written", stats)
