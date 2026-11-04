# LLM-assisted traffic planning: can LLMs formulate and solve user-equilibrium problems reliably?

This project measures whether a large language model can turn a plain-English traffic planning
request (close a road, add capacity, introduce a toll, change demand) into a correct formulation
that a classical traffic-assignment solver can run, and how much verification improves that.

## Method
- **Network:** Sioux Falls benchmark (TNTP), 24 nodes, 76 directed links, 528 OD pairs.
- **Reference solver:** Static user equilibrium via Frank–Wolfe with BPR link costs.
  Validated against published equilibrium link flows (mean relative error 0.008%, max 0.04%).
- **Benchmark:** 40 scenarios with solver-computed reference answers (`experiments/scenarios.json`).
- **Conditions:**
  - A. Direct: the LLM estimates the resulting total system travel time (TSTT) without a solver.
  - B. Formulate: the LLM emits a structured JSON spec; the solver computes TSTT.
  - C. Verify: as B, with constraint checks and up to two retries using the error messages.
- **Metrics:** formulation correctness, TSTT relative error vs reference, retry counts, error types.

## Reproduce
```bash
pip install -r requirements.txt
python scripts/validate_reference.py   # solver check against published flows
python scripts/build_scenarios.py      # regenerate the scenario benchmark
pytest -q                              # unit tests (fast)
```

## Limitations
- Structured JSON specs replace free-form optimization models. This isolates whether the LLM
  picks the right intervention, not whether it writes correct solver code.
- One network and one model family per run. Results are not claimed to generalize.
- Static assignment only; no dynamic or multimodal effects.
