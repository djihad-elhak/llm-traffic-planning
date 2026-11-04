"""Planning scenarios: a plain-English request paired with a machine-checkable reference answer."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .network import Network, Link
from .assignment import solve_user_equilibrium


@dataclass
class Scenario:
    sid: str
    text: str                       # what a planner would say, in English
    op: str                         # close_link | capacity | toll | demand
    params: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, float] = field(default_factory=dict)


def apply_scenario(net: Network, sc: Scenario) -> tuple[Network, np.ndarray | None]:
    """Return the modified network and optional per-link routing toll vector."""
    new = copy.deepcopy(net)
    tolls = None
    if sc.op == "close_link":
        i, j = sc.params["init"], sc.params["term"]
        new.links = [l for l in new.links if not (l.init == i and l.term == j)]
    elif sc.op == "capacity":
        i, j, f = sc.params["init"], sc.params["term"], sc.params["factor"]
        for l in new.links:
            if (l.init, l.term) == (i, j):
                l.capacity *= f
    elif sc.op == "toll":
        i, j, amt = sc.params["init"], sc.params["term"], sc.params["amount"]
        tolls = np.zeros(len(new.links))
        for idx, l in enumerate(new.links):
            if (l.init, l.term) == (i, j):
                tolls[idx] = amt
    elif sc.op == "demand":
        f = sc.params["factor"]
        new.demand = {k: v * f for k, v in new.demand.items()}
    else:
        raise ValueError(f"Unknown op {sc.op}")
    return new, tolls


def is_connected(net: Network) -> bool:
    """Every OD pair must remain routable."""
    from .assignment import _shortest_path_trees  # local import to avoid cycles
    costs = np.array([l.free_flow_time for l in net.links])
    _, dists = _shortest_path_trees(net, costs)
    return all(np.isfinite(dists[o][d]) for (o, d) in net.demand)


def reference_metrics(base: Network, sc: Scenario, base_tstt: float) -> dict[str, float]:
    """Solve the modified network and report TSTT and its change from baseline."""
    new_net, tolls = apply_scenario(base, sc)
    res = solve_user_equilibrium(new_net, max_iter=5000, tol=1e-5, extra_costs=tolls)
    return {
        "baseline_tstt": base_tstt,
        "scenario_tstt": res.tstt,
        "tstt_change_pct": 100.0 * (res.tstt - base_tstt) / base_tstt,
        "relative_gap": res.relative_gap,
    }


def build_scenarios(base: Network, n: int = 40, seed: int = 7, compute_reference: bool = True) -> list[Scenario]:
    """Deterministic generation: pick links whose removal keeps the network connected."""
    rng = np.random.default_rng(seed)
    candidates: list[Scenario] = []
    links = base.links
    order = rng.permutation(len(links))

    kinds = ["close_link", "capacity", "toll", "capacity", "toll", "demand"]
    k = 0
    for idx in order:
        l: Link = links[idx]
        kind = kinds[k % len(kinds)]
        k += 1
        if kind == "close_link":
            probe, _ = apply_scenario(base, Scenario("p", "", "close_link", {"init": l.init, "term": l.term}))
            if not is_connected(probe):
                continue
            text = f"Close the road segment from node {l.init} to node {l.term} for roadworks."
            params = {"init": l.init, "term": l.term}
        elif kind == "capacity":
            factor = float(rng.choice([1.5, 2.0]))
            text = f"Widen the link from node {l.init} to node {l.term} so its capacity increases by a factor of {factor:g}."
            params = {"init": l.init, "term": l.term, "factor": factor}
        elif kind == "toll":
            amount = float(rng.choice([5.0, 10.0, 20.0]))
            text = f"Introduce a toll of {amount:g} on the link from node {l.init} to node {l.term}."
            params = {"init": l.init, "term": l.term, "amount": amount}
        else:
            factor = float(rng.choice([0.9, 1.1, 1.2]))
            text = f"Scale total travel demand across the whole network by {factor:g}."
            params = {"factor": factor}
            kind = "demand"
            if sum(1 for c in candidates if c.op == "demand") >= 4:
                continue
        candidates.append(Scenario(sid="", text=text, op=kind, params=params))
        if len(candidates) >= n:
            break

    for i, sc in enumerate(candidates, 1):
        sc.sid = f"S{i:02d}"
    if compute_reference:
        base_tstt = solve_user_equilibrium(base, max_iter=5000, tol=1e-5).tstt
        for sc in candidates:
            sc.reference = reference_metrics(base, sc, base_tstt)
    return candidates


def scenario_to_dict(sc: Scenario) -> dict:
    return asdict(sc)
