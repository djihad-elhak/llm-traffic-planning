"""Static user-equilibrium traffic assignment via Frank-Wolfe.

Link cost is the BPR function t = t0 * (1 + b * (v / c) ** power).
The objective is the Beckmann function, minimized by Frank-Wolfe with
an exact line search (bisection on the directional derivative).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .network import Network


@dataclass
class AssignmentResult:
    flows: np.ndarray          # link flows, aligned with network.links
    link_times: np.ndarray     # congested link travel times
    tstt: float                # total system travel time
    relative_gap: float        # convergence measure
    iterations: int


def bpr_time(net: Network, flows: np.ndarray) -> np.ndarray:
    t0 = np.array([l.free_flow_time for l in net.links])
    cap = np.array([l.capacity for l in net.links])
    b = np.array([l.b for l in net.links])
    p = np.array([l.power for l in net.links])
    return t0 * (1.0 + b * (flows / cap) ** p)


def _bpr_integral(net: Network, flows: np.ndarray) -> float:
    t0 = np.array([l.free_flow_time for l in net.links])
    cap = np.array([l.capacity for l in net.links])
    b = np.array([l.b for l in net.links])
    p = np.array([l.power for l in net.links])
    return float(np.sum(t0 * (flows + b * cap * ((flows / cap) ** (p + 1)) / (p + 1))))


def _shortest_path_trees(net: Network, costs: np.ndarray):
    """Dijkstra from every zone. Returns predecessor-link arrays per origin."""
    adj: dict[int, list[tuple[int, float, int]]] = {n: [] for n in range(1, net.num_nodes + 1)}
    for idx, l in enumerate(net.links):
        adj[l.init].append((l.term, costs[idx], idx))

    trees = {}
    dist_all = {}
    for origin in net.zones:
        dist = {n: float("inf") for n in adj}
        pred_link = {n: -1 for n in adj}
        dist[origin] = 0.0
        heap = [(0.0, origin)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, c, idx in adj[u]:
                nd = d + c
                if nd < dist[v]:
                    dist[v] = nd
                    pred_link[v] = idx
                    heapq.heappush(heap, (nd, v))
        trees[origin] = pred_link
        dist_all[origin] = dist
    return trees, dist_all


def _all_or_nothing(net: Network, costs: np.ndarray) -> np.ndarray:
    trees, _ = _shortest_path_trees(net, costs)
    aux = np.zeros(len(net.links))
    for (origin, dest), trips in net.demand.items():
        node = dest
        pred = trees[origin]
        while node != origin:
            idx = pred[node]
            if idx < 0:
                raise ValueError(f"No path from {origin} to {dest}")
            aux[idx] += trips
            node = net.links[idx].init
    return aux


def _line_search(net: Network, flows: np.ndarray, aux: np.ndarray) -> float:
    lo, hi = 0.0, 1.0
    direction = aux - flows

    def deriv(alpha: float) -> float:
        f = flows + alpha * direction
        return float(np.dot(bpr_time(net, f), direction))

    if deriv(1.0) <= 0:
        return 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if deriv(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve_user_equilibrium(
    net: Network,
    max_iter: int = 1000,
    tol: float = 1e-6,
    extra_costs: np.ndarray | None = None,
) -> AssignmentResult:
    """Solve static UE. `extra_costs` adds per-link monetary/penalty terms to
    the travel time used for routing (e.g. tolls), but not to reported TSTT."""
    n = len(net.links)
    flows = _all_or_nothing(net, bpr_time(net, np.zeros(n)))
    gap = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        times = bpr_time(net, flows)
        route_costs = times + (extra_costs if extra_costs is not None else 0.0)
        aux = _all_or_nothing(net, route_costs)
        sptt = float(np.dot(times, flows))
        dtt = float(np.dot(times, aux))
        gap = (sptt - dtt) / sptt if sptt > 0 else 0.0
        if gap < tol:
            break
        alpha = _line_search(net, flows, aux)
        flows = flows + alpha * (aux - flows)

    times = bpr_time(net, flows)
    tstt = float(np.dot(times, flows))
    return AssignmentResult(flows=flows, link_times=times, tstt=tstt, relative_gap=gap, iterations=it)
