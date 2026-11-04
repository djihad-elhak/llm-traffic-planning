"""Three experimental conditions for LLM-assisted traffic planning.

A  direct    : the LLM is asked for the resulting total system travel time (TSTT) directly.
B  formulate : the LLM converts the request into a structured spec; the solver computes TSTT.
C  verify    : as B, but the spec is checked against network constraints and the LLM
               retries with the error message (up to MAX_RETRIES).

The spec is a JSON object, not a free-form Pyomo model. This keeps the formulation
checkable against the reference scenario parameters.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .network import Network
from .assignment import solve_user_equilibrium
from .scenarios import Scenario, apply_scenario, is_connected

MAX_RETRIES = 2
VALID_OPS = {"close_link", "capacity", "toll", "demand"}

SPEC_INSTRUCTIONS = """You translate a traffic planning request into a JSON spec.
The network is Sioux Falls (24 nodes, 76 directed links). Output ONLY a JSON object:
{"op": "close_link" | "capacity" | "toll" | "demand",
 "init": <int node, for link ops>, "term": <int node, for link ops>,
 "factor": <number, for capacity and demand>, "amount": <number, for toll>}
Use only the keys relevant to the op. No prose."""


@dataclass
class Trial:
    sid: str
    condition: str
    model: str
    raw_output: str = ""
    parsed: dict[str, Any] | None = None
    predicted_tstt: float | None = None
    formulation_correct: bool = False
    tstt_rel_error: float | None = None
    attempts: int = 1
    errors: list[str] = field(default_factory=list)


class LLMClient:
    """Thin wrapper so the harness can be tested without network access."""

    def __init__(self, model: str):
        import anthropic  # imported lazily so unit tests don't need the SDK
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str, max_tokens: int = 300) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def parse_json_object(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def validate_spec(net: Network, spec: dict[str, Any]) -> list[str]:
    """Return a list of constraint violations; empty means valid."""
    errs: list[str] = []
    op = spec.get("op")
    if op not in VALID_OPS:
        return [f"op must be one of {sorted(VALID_OPS)}, got {op!r}"]
    if op in {"close_link", "capacity", "toll"}:
        pair = (spec.get("init"), spec.get("term"))
        if pair not in net.link_index():
            errs.append(f"no link from node {pair[0]} to node {pair[1]} in the network")
    if op in {"capacity", "demand"} and not isinstance(spec.get("factor"), (int, float)):
        errs.append("factor must be a number")
    if op == "toll" and not isinstance(spec.get("amount"), (int, float)):
        errs.append("amount must be a number")
    if op == "close_link" and not errs:
        sc = Scenario("v", "", "close_link", {"init": spec["init"], "term": spec["term"]})
        probe, _ = apply_scenario(net, sc)
        if not is_connected(probe):
            errs.append("closing this link disconnects an origin-destination pair")
    return errs


def spec_to_scenario(spec: dict[str, Any]) -> Scenario:
    op = spec["op"]
    if op == "toll":
        params = {"init": spec["init"], "term": spec["term"], "amount": float(spec["amount"])}
    elif op == "capacity":
        params = {"init": spec["init"], "term": spec["term"], "factor": float(spec["factor"])}
    elif op == "close_link":
        params = {"init": spec["init"], "term": spec["term"]}
    else:
        params = {"factor": float(spec["factor"])}
    return Scenario("pred", "", op, params)


def spec_matches_reference(spec: dict[str, Any], ref: Scenario) -> bool:
    if spec.get("op") != ref.op:
        return False
    for key, val in ref.params.items():
        got = spec.get(key)
        if got is None:
            return False
        if isinstance(val, float):
            if abs(float(got) - val) > 1e-9:
                return False
        elif int(got) != int(val):
            return False
    return True


def run_trial(net: Network, sc: Scenario, condition: str, client: LLMClient) -> Trial:
    trial = Trial(sid=sc.sid, condition=condition, model=client.model)

    if condition == "A":
        prompt = (
            "Sioux Falls network, user equilibrium with BPR costs (alpha=0.15, beta=4). "
            f"Planning request: {sc.text}\n"
            "Give only the resulting total system travel time as a single number."
        )
        trial.raw_output = client.complete("You are an expert transportation modeller.", prompt, max_tokens=50)
        nums = re.findall(r"[\d,]+\.?\d*", trial.raw_output)
        if nums:
            trial.predicted_tstt = float(nums[0].replace(",", ""))
        return trial

    feedback = ""
    for attempt in range(1, MAX_RETRIES + 2 if condition == "C" else 2):
        trial.attempts = attempt
        user = f"Request: {sc.text}" + (f"\nYour previous spec was rejected: {feedback}\nFix it." if feedback else "")
        trial.raw_output = client.complete(SPEC_INSTRUCTIONS, user)
        spec = parse_json_object(trial.raw_output)
        if spec is None:
            feedback = "output was not a JSON object"
            trial.errors.append(feedback)
            continue
        trial.parsed = spec
        if condition == "C":
            errs = validate_spec(net, spec)
            if errs:
                feedback = "; ".join(errs)
                trial.errors.append(feedback)
                continue
        break

    if trial.parsed is None:
        return trial
    trial.formulation_correct = spec_matches_reference(trial.parsed, sc)
    try:
        pred_sc = spec_to_scenario(trial.parsed)
        if validate_spec(net, trial.parsed):
            return trial
        new_net, tolls = apply_scenario(net, pred_sc)
        res = solve_user_equilibrium(new_net, max_iter=5000, tol=1e-5, extra_costs=tolls)
        trial.predicted_tstt = res.tstt
    except Exception as exc:  # solver or spec errors are data, not crashes
        trial.errors.append(f"solver error: {exc}")
    return trial


class HFClient:
    """Open-weight instruct model on a local GPU (e.g. Kaggle T4), 4-bit quantized.

    Requires: transformers, accelerate, bitsandbytes. Free on Kaggle's T4 GPUs.
    """

    def __init__(self, model: str, max_new_tokens: int = 300):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model = model
        self.max_new_tokens = max_new_tokens
        self._tok = AutoTokenizer.from_pretrained(model)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=quant, device_map="auto"
        )
        self._torch = torch

    def complete(self, system: str, user: str, max_tokens: int = 300) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, self.max_new_tokens),
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
