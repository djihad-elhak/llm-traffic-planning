"""Generate llm_experiments.ipynb (Kaggle notebook). Run: python scripts/make_notebook.py"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# LLM-assisted traffic planning: experiment notebook

**Kaggle setup (required):** Settings → Accelerator → **GPU T4 x2**, and Settings → Internet → **On**.

**What this notebook does**
1. Verifies the solver and benchmark (sanity checks).
2. Runs conditions A (direct estimate), B (structured formulation), and C (formulation with verification and retries) on every benchmark scenario, for each model in `MODELS`.
3. Classifies every formulation error, records per-trial timing, and saves run metadata.
4. Produces summary tables and figures in `/kaggle/working/`.

Runs are resumable: each trial is written to disk as soon as it finishes, so an interrupted session can be restarted and will skip completed trials.""")

code("""# 0. Environment and GPU check (skipped in mock-test mode)
import os
MOCK = os.environ.get("NB_MOCK") == "1"
if not MOCK:
    !nvidia-smi --query-gpu=name,memory.total --format=csv""")

code("""# 1. Install extras (Kaggle preinstalls torch)
if not MOCK:
    %pip install -q transformers accelerate bitsandbytes""")

code("""# 2. Get the code at a known commit
import os, sys, subprocess
if not os.path.isdir("src"):
    if not os.path.isdir("llm-traffic-planning"):
        !git clone -q https://github.com/djihad-elhak/llm-traffic-planning
    else:
        !git -C llm-traffic-planning pull -q
    os.chdir("llm-traffic-planning")
sys.path.insert(0, os.getcwd())
GIT_COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
print("working directory:", os.getcwd(), "| commit:", GIT_COMMIT)""")

code("""# 3. Configuration
import json, time, dataclasses, platform
from pathlib import Path
import numpy as np

SEED = 0
np.random.seed(SEED)
# Primary model, and a larger model for the scale comparison. Each model is loaded, run, then freed.
MODELS = ["mock"] if MOCK else ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]
CONDITIONS = ["A", "B", "C"]
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUFFIX = "mock" if MOCK else "kaggle"
print("mock mode:", MOCK, "| models:", MODELS, "| output:", OUT_DIR)""")

code("""# 4. Sanity checks: data loads, scenarios load, and the solver reproduces the validated baseline
from src.network import load_network
from src.assignment import solve_user_equilibrium
from src.scenarios import Scenario

DATA = Path("data")
net = load_network(DATA / "SiouxFalls_net.tntp", DATA / "SiouxFalls_trips.tntp", "SiouxFalls")
scenarios = [Scenario(**{k: v for k, v in d.items() if k in {"sid", "text", "op", "params", "reference"}})
             for d in json.loads(Path("experiments/scenarios.json").read_text())]
SCEN_BY_ID = {s.sid: s for s in scenarios}
base = solve_user_equilibrium(net, max_iter=5000, tol=1e-5)
print(f"links={len(net.links)}  OD pairs={len(net.demand)}  scenarios={len(scenarios)}")
print(f"baseline TSTT={base.tstt:,.0f}  relative gap={base.relative_gap:.1e}")
assert len(scenarios) == 40
assert abs(base.tstt - 7480158) / 7480158 < 1e-3, "baseline does not match the validated value"
print("sanity checks passed")""")

code("""# 5. Run metadata, saved next to the results so every run can be reproduced
import importlib.metadata as md_
def version(pkg):
    try:
        return md_.version(pkg)
    except md_.PackageNotFoundError:
        return "not installed"
gpu = "none (mock)"
if not MOCK:
    import torch
    gpu = torch.cuda.get_device_name(0) + f" x{torch.cuda.device_count()}"
META = {
    "git_commit": GIT_COMMIT,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "gpu": gpu,
    "models": MODELS,
    "conditions": CONDITIONS,
    "seed": SEED,
    "scenarios": len(scenarios),
    "versions": {p: version(p) for p in ["numpy", "scipy", "torch", "transformers", "accelerate", "bitsandbytes"]},
}
(OUT_DIR / f"run_metadata_{SUFFIX}.json").write_text(json.dumps(META, indent=2))
print(json.dumps(META, indent=2))""")

code("""# 6. Helpers: model loading and error classification
from src.llm_conditions import HFClient, MockClient, run_trial
link_idx = net.link_index()

def load_client(name):
    if name == "mock":
        return MockClient(scenarios)
    import torch
    assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU T4 x2"
    client = HFClient(name)
    print(f"loaded {name} | GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return client

def classify_error(row, sc):
    \"\"\"Label a formulation as correct, or name the kind of error.\"\"\"
    if row["formulation_correct"]:
        return "correct"
    spec = row["parsed"]
    if spec is None:
        return "unparsable"
    if spec.get("op") != sc.op:
        return "wrong_operation"
    if sc.op in {"close_link", "capacity", "toll"}:
        pair = (spec.get("init"), spec.get("term"))
        if pair not in link_idx:
            return "nonexistent_link"
        if pair != (sc.params["init"], sc.params["term"]):
            return "wrong_link"
    return "wrong_magnitude"
print("helpers ready")""")

code("""# 7. Run every model: smoke test, then the full benchmark (resumable)
def run_model(name):
    client = load_client(name)
    results = OUT_DIR / f"results_{name.replace('/', '_')}_{SUFFIX}.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            done.add((r["sid"], r["condition"]))
    # smoke test on one scenario
    for cond in CONDITIONS:
        t = run_trial(net, scenarios[0], cond, client)
        print(f"  smoke {cond}: correct={t.formulation_correct} predicted={t.predicted_tstt}")
    with results.open("a") as fh:
        for sc in scenarios:
            for cond in CONDITIONS:
                if (sc.sid, cond) in done:
                    continue
                t0 = time.time()
                trial = run_trial(net, sc, cond, client)
                row = dataclasses.asdict(trial)
                row["elapsed_s"] = round(time.time() - t0, 3)
                row["reference_tstt"] = sc.reference["scenario_tstt"]
                row["reference_change_pct"] = sc.reference["tstt_change_pct"]
                row["error_type"] = classify_error(row, sc) if cond != "A" else None
                fh.write(json.dumps(row) + "\\n")
                fh.flush()
    print(f"{name}: results in {results}")
    del client
    if not MOCK:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    return results

RESULT_FILES = {}
for m in MODELS:
    print("=== model:", m)
    RESULT_FILES[m] = run_model(m)""")

code("""# 8. Summary per model and condition
import pandas as pd
rng = np.random.default_rng(SEED)

def load_results(path):
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines()])

rows = []
for m, path in RESULT_FILES.items():
    df = load_results(path)
    df["pred_err_pct"] = 100 * (df["predicted_tstt"] - df["reference_tstt"]).abs() / df["reference_tstt"]
    for cond, g in df.groupby("condition"):
        rec = {"model": m, "condition": cond, "n": len(g),
               "no_prediction": int(g["predicted_tstt"].isna().sum()),
               "median_tstt_err_pct": round(g["pred_err_pct"].median(), 2),
               "p90_tstt_err_pct": round(g["pred_err_pct"].quantile(0.9), 2),
               "mean_attempts": round(g["attempts"].mean(), 2),
               "mean_trial_seconds": round(g["elapsed_s"].mean(), 2),
               "total_minutes": round(g["elapsed_s"].sum() / 60, 1)}
        if cond != "A":
            x = g["formulation_correct"].values.astype(float)
            boots = [rng.choice(x, len(x)).mean() for _ in range(2000)]
            rec["formulation_acc"] = round(x.mean(), 3)
            rec["acc_95ci_low"] = round(np.percentile(boots, 2.5), 3)
            rec["acc_95ci_high"] = round(np.percentile(boots, 97.5), 3)
        rows.append(rec)
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / f"summary_{SUFFIX}.csv", index=False)
summary""")

code("""# 9. Error taxonomy: what kinds of formulation errors occurred (conditions B and C)
tax = []
for m, path in RESULT_FILES.items():
    df = load_results(path)
    d = df[df["condition"].isin(["B", "C"])]
    counts = d.groupby(["condition", "error_type"]).size().unstack(fill_value=0)
    counts.insert(0, "model", m)
    tax.append(counts.reset_index())
taxonomy = pd.concat(tax, ignore_index=True)
taxonomy.to_csv(OUT_DIR / f"error_taxonomy_{SUFFIX}.csv", index=False)
taxonomy""")

code("""# 10. Figures: formulation accuracy with 95% CI, and TSTT error by condition
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
acc = summary[summary["condition"].isin(["B", "C"])].copy()
labels = [f"{r.model.split('/')[-1]}\\n{r.condition}" for r in acc.itertuples()]
yerr = np.vstack([acc["formulation_acc"] - acc["acc_95ci_low"], acc["acc_95ci_high"] - acc["formulation_acc"]])
axes[0].bar(labels, acc["formulation_acc"], yerr=yerr, capsize=4, color="#1f4e79")
axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Formulation accuracy")
axes[0].set_title("Formulation accuracy (95% bootstrap CI)")
errs = []
names = []
for m, path in RESULT_FILES.items():
    df = load_results(path)
    df["pred_err_pct"] = 100 * (df["predicted_tstt"] - df["reference_tstt"]).abs() / df["reference_tstt"]
    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, "pred_err_pct"].dropna().values
        if len(vals):
            errs.append(vals); names.append(f"{m.split('/')[-1]}\\n{cond}")
axes[1].boxplot(errs, tick_labels=names, showfliers=False)
axes[1].set_yscale("symlog", linthresh=0.01); axes[1].set_ylabel("TSTT error (%), symlog scale")
axes[1].set_title("Relative TSTT error, predicted scenarios")
for ax in axes: ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(OUT_DIR / f"figures_{SUFFIX}.png", dpi=150)
plt.show()""")

code("""# 11. Final report: total run time and where it went
total_min = summary["total_minutes"].sum()
print(f"Total trial time (solver + model): {total_min:.1f} minutes")
print(summary[["model", "condition", "mean_trial_seconds", "total_minutes"]].to_string(index=False))
print("Files written to", OUT_DIR, ":", sorted(p.name for p in OUT_DIR.iterdir()))""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nbf.write(nb, ROOT / "llm_experiments.ipynb")
print("wrote llm_experiments.ipynb with", len(cells), "cells")
