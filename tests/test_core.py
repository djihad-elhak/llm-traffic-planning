"""Fast unit tests. Run with: pytest -q"""

from pathlib import Path

import numpy as np
import pytest

from src.network import load_network
from src.assignment import bpr_time, solve_user_equilibrium
from src.llm_conditions import parse_json_object, validate_spec, spec_matches_reference
from src.scenarios import Scenario

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def net():
    return load_network(DATA / "SiouxFalls_net.tntp", DATA / "SiouxFalls_trips.tntp", "SiouxFalls")


def test_network_dimensions(net):
    assert len(net.links) == 76
    assert net.num_nodes == 24
    assert sum(net.demand.values()) == pytest.approx(360600)


def test_bpr_free_flow_at_zero_flow(net):
    t = bpr_time(net, np.zeros(len(net.links)))
    assert np.allclose(t, [l.free_flow_time for l in net.links])


def test_bpr_increases_with_flow(net):
    low = bpr_time(net, np.full(len(net.links), 1000.0))
    high = bpr_time(net, np.full(len(net.links), 50000.0))
    assert np.all(high > low)


def test_parse_json_object_extracts_from_prose():
    assert parse_json_object('Sure: {"op": "demand", "factor": 1.1} done') == {"op": "demand", "factor": 1.1}


def test_parse_json_object_rejects_garbage():
    assert parse_json_object("no json here") is None
    assert parse_json_object("{not valid") is None


def test_validate_rejects_unknown_link(net):
    errs = validate_spec(net, {"op": "toll", "init": 1, "term": 24, "amount": 5})
    assert any("no link" in e for e in errs)


def test_validate_rejects_bad_op(net):
    assert validate_spec(net, {"op": "teleport"})


def test_validate_accepts_good_toll(net):
    assert validate_spec(net, {"op": "toll", "init": 1, "term": 2, "amount": 5}) == []


def test_validate_accepts_connected_closure(net):
    assert validate_spec(net, {"op": "close_link", "init": 1, "term": 2}) == []


def test_spec_matches_reference():
    ref = Scenario("S01", "", "capacity", {"init": 1, "term": 2, "factor": 1.5})
    assert spec_matches_reference({"op": "capacity", "init": 1, "term": 2, "factor": 1.5}, ref)
    assert not spec_matches_reference({"op": "capacity", "init": 1, "term": 2, "factor": 2.0}, ref)
    assert not spec_matches_reference({"op": "toll", "init": 1, "term": 2, "amount": 5}, ref)


def test_equilibrium_gap_small_after_solve(net):
    res = solve_user_equilibrium(net, max_iter=5000, tol=1e-4)
    assert res.relative_gap < 1e-3
    assert res.tstt > 0
