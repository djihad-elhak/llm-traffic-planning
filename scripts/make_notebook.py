"""Generate llm_experiments.ipynb (Kaggle notebook). Run: python scripts/make_notebook.py"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# LLM-assisted traffic planning: experiment notebook

**Kaggle setup (required):**
1. Settings → Accelerator → **GPU T4 x2**
2. Settings → Internet → **On** (needed to clone the repo and download the model)
3. Run all cells. Results are saved to `/kaggle/working/` and survive if the session stops; re-running resumes where it left off.

Conditions: **A** direct answer · **B** structured spec, solver computes · **C** spec verified against constraints with retries.""")

code("""# 0. GPU check (skipped automatically in mock-test mode)
import os
MOCK = os.environ.get("NB_MOCK") == "1"
if not MOCK:
    !nvidia-smi --query-gpu=name,memory.total --format=csv""")

code("""# 1. Install (Kaggle has torch preinstalled; these are the extras)
if not MOCK:
    %pip install -q transformers accelerate bitsandbytes""")

code("""# 2. Get the code
import os, sys
if not os.path.isdir("src"):
    if not os.path.isdir("llm-traffic-planning"):
        !git clone -q https://github.com/djihad-elhak/llm-traffic-planning
    else:
        !git -C llm-traffic-planning pull -q
    os.chdir("llm-traffic-planning")
sys.path.insert(0, os.getcwd())
print("working directory:", os.getcwd())""")

code("""# 3. Configuration
from pathlib import Path
MODEL = "Qwen/Qwen2.5-7B-Instruct"   # open-weight, 4-bit fits a T4
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = OUT_DIR / ("results_mock.jsonl" if MOCK else "results_qwen7b.jsonl")
print("mock mode:", MOCK, "| results file:", RESULTS)""")

code("""# 4. Sanity checks: data loads and the solver reproduces the baseline
import json
from src.network import load_network
from src.assignment import solve_user_equilibrium
from src.scenarios import Scenario

DATA = Path("data")
net = load_network(DATA / "SiouxFalls_net.tntp", DATA / "SiouxFalls_trips.tntp", "SiouxFalls")
scenarios = [Scenario(**{k: v for k, v in d.items() if k in {"sid", "text", "op", "params", "reference"}})
             for d in json.loads(Path("experiments/scenarios.json").read_text())]
base = solve_user_equilibrium(net, max_iter=5000, tol=1e-5)
print(f"links={len(net.links)}  OD pairs={len(net.demand)}  scenarios={len(scenarios)}")
print(f"baseline TSTT={base.tstt:,.0f}  gap={base.relative_gap:.1e}")
assert len(scenarios) == 40
assert abs(base.tstt - 7480158) / 7480158 < 1e-3, "baseline does not match validated value"
print("sanity checks passed")""")

code("""# 5. Load the model (or the mock)
from src.llm_conditions import HFClient, MockClient, run_trial
if MOCK:
    client = MockClient(scenarios)
else:
    import torch
    assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU T4 x2"
    client = HFClient(MODEL)
    print(f"model loaded | GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f} GB")""")

code("""# 6. Smoke test on one scenario, all conditions
for cond in ["A", "B", "C"]:
    t = run_trial(net, scenarios[0], cond, client)
    print(cond, "| formulation correct:", t.formulation_correct,
          "| predicted TSTT:", t.predicted_tstt, "| attempts:", t.attempts)
    print("   raw:", t.raw_output[:120].replace(chr(10), " "))""")

code("""# 7. Full run (resumable: finished scenario/condition pairs are skipped)
import dataclasses
done = set()
if RESULTS.exists():
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        done.add((r["sid"], r["condition"]))
with RESULTS.open("a") as fh:
    for sc in scenarios:
        for cond in ["A", "B", "C"]:
            if (sc.sid, cond) in done:
                continue
            trial = run_trial(net, sc, cond, client)
            row = dataclasses.asdict(trial)
            row["reference_tstt"] = sc.reference["scenario_tstt"]
            row["reference_change_pct"] = sc.reference["tstt_change_pct"]
            fh.write(json.dumps(row) + "\\n")
            fh.flush()
print("done. rows:", sum(1 for _ in RESULTS.open()))""")

code("""# 8. Summary: formulation accuracy, TSTT error, retries
import pandas as pd, numpy as np
df = pd.DataFrame([json.loads(l) for l in RESULTS.read_text().splitlines()])
df["pred_err_pct"] = 100 * (df["predicted_tstt"] - df["reference_tstt"]).abs() / df["reference_tstt"]
rows = []
for cond, g in df.groupby("condition"):
    acc = g["formulation_correct"].mean()
    # bootstrap 95% CI for formulation accuracy
    rng = np.random.default_rng(0)
    boots = [rng.choice(g["formulation_correct"].values.astype(float), len(g)).mean() for _ in range(2000)]
    rows.append({
        "condition": cond,
        "n": len(g),
        "formulation_acc": round(acc, 3) if cond != "A" else None,
        "acc_95ci": (f"[{np.percentile(boots,2.5):.2f}, {np.percentile(boots,97.5):.2f}]" if cond != "A" else None),
        "no_prediction": int(g["predicted_tstt"].isna().sum()),
        "median_tstt_err_pct (predicted only)": round(g["pred_err_pct"].median(), 2),
        "p90_tstt_err_pct (predicted only)": round(g["pred_err_pct"].quantile(0.9), 2),
        "mean_attempts": round(g["attempts"].mean(), 2),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / ("summary_mock.csv" if MOCK else "summary_qwen7b.csv"), index=False)
summary""")

code("""# 9. Errors: which kinds of mistakes happened (conditions B and C)
errs = df[df["condition"].isin(["B", "C"])]
print("trials with any recorded error:", int((errs["errors"].str.len() > 0).sum()), "of", len(errs))
print(errs.loc[~errs["formulation_correct"], ["sid", "condition", "parsed", "errors"]].head(15).to_string())""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nbf.write(nb, ROOT / "llm_experiments.ipynb")
print("wrote llm_experiments.ipynb with", len(cells), "cells")
