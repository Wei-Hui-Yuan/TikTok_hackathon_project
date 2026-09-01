"""Autonomous ML research agent for KuaiRand-Pure.

Each iteration, an LLM (OpenAI Responses API) is shown the full experiment
history and asked to propose the next experiment across three real pipeline
levers: model/loss (fm=pointwise BCE, bpr=pairwise BPR), feature engineering
(which optional item/user fields to add on top of the fixed 5-field base),
and hyperparameters (lr, k, seed) — along with a one-sentence hypothesis for
why. The proposal is validated against a whitelist before it's ever used to
build a subprocess command (LLM output is treated as untrusted input, not
executed directly). If the OpenAI call is unavailable, returns invalid JSON
twice in a row, or proposes a config already tried, the agent falls back to a
deterministic heuristic so the run never stalls on an external dependency.

Stopping follows the challenge's convergence rule: converged when the
validation-best primary hasn't improved by more than CONVERGENCE_EPS over
the last CONVERGENCE_N successful iterations, or the iteration/wall-clock
budget is exhausted, whichever comes first.
"""
import json
import os
import re
import subprocess
import sys
import time

from data import EXTRA_FIELD_CHOICES

MAX_ITERATIONS = 50
WALL_CLOCK_LIMIT_S = 6 * 60 * 60
EXPERIMENT_TIMEOUT_S = 30 * 60
CONVERGENCE_EPS = 0.002
CONVERGENCE_N = 3

LOG_FILE = "experiment_log.json"
SUMMARY_FILE = "experiment_summary.json"

MODEL_CHOICES = ["fm", "bpr"]
OPENAI_MODEL = os.environ.get("AGENT_OPENAI_MODEL", "gpt-5.6")

PLANNER_INSTRUCTIONS = f"""You are the research-decision brain of an autonomous ML research agent for \
the KuaiRand-Pure short-video recommendation benchmark.

Task: within-user ranking of logged impressions. Label: long_view (binary). \
Metric to maximize: validation primary = mean(GAUC, nDCG@5).

You control three real levers of the pipeline (not just hyperparameters):
- model: one of {MODEL_CHOICES} — "fm" is pointwise binary cross-entropy, \
"bpr" is pairwise Bayesian Personalized Ranking loss on the same Factorization \
Machine.
- extra_fields: a list, any subset of {EXTRA_FIELD_CHOICES} (possibly empty). \
These are optional item-side and user-side categorical fields added on top of \
the fixed 5-field base (user_id, video_id, author_id, tab, dur_bucket).
- lr (float), k (int, FM embedding dimension), seed (int).

You will be shown the full history of experiments tried so far, each with its \
config, your (or a previous run's) stated hypothesis, and the resulting \
validation GAUC/nDCG@5/primary (or the failure reason if it did not complete).

Propose exactly one new experiment to try next, and explain briefly why, given \
what has and hasn't worked so far. Do not repeat a configuration already tried.

Reply with ONLY a single JSON object and nothing else — no markdown code \
fences, no prose before or after it:
{{"model": "fm|bpr", "extra_fields": ["..."], "lr": 0.001, "k": 16, "seed": 0, \
"hypothesis": "one or two sentences on what you are testing and why"}}
"""

RETRY_SUFFIX = "\n\nYour previous reply was not a single valid JSON object. Reply with ONLY the JSON object, nothing else."


# ==========================================
# MEMORY
# ==========================================

def load_history():
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_history(history):
    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=4)


# ==========================================
# LLM-DRIVEN PLANNER
# ==========================================

def summarize_history(history):
    if not history:
        return "(no experiments run yet)"
    lines = []
    for x in history:
        cfg = (f"model={x.get('model')} extra_fields={x.get('extra_fields', [])} "
               f"lr={x.get('lr')} k={x.get('k')} seed={x.get('seed')}")
        if x.get("status") == "success":
            lines.append(
                f"#{x['experiment_number']} [{x.get('planner', '?')}] {cfg}\n"
                f"  hypothesis: {x.get('hypothesis', x.get('reason', ''))}\n"
                f"  result: GAUC={x['valid_gauc']:.4f} nDCG@5={x['valid_ndcg']:.4f} "
                f"primary={x['valid_primary']:.4f}"
            )
        else:
            lines.append(
                f"#{x['experiment_number']} [{x.get('planner', '?')}] {cfg}\n"
                f"  hypothesis: {x.get('hypothesis', x.get('reason', ''))}\n"
                f"  result: FAILED ({x.get('failure_reason', 'unknown')})"
            )
    return "\n".join(lines)


def parse_json_object(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text.strip(), re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_proposal(raw):
    """Whitelist-validate the LLM's proposal before it ever reaches a subprocess call."""
    if not isinstance(raw, dict):
        return None, "not a JSON object"

    model = raw.get("model")
    if model not in MODEL_CHOICES:
        return None, f"model must be one of {MODEL_CHOICES}"

    extra_fields = raw.get("extra_fields", [])
    if not isinstance(extra_fields, list) or not all(isinstance(f, str) for f in extra_fields):
        return None, "extra_fields must be a list of strings"
    unknown = [f for f in extra_fields if f not in EXTRA_FIELD_CHOICES]
    if unknown:
        return None, f"unknown extra_fields {unknown}, choices are {EXTRA_FIELD_CHOICES}"

    try:
        lr = float(raw["lr"])
        k = int(raw["k"])
        seed = int(raw.get("seed", 0))
    except (KeyError, TypeError, ValueError):
        return None, "lr/k/seed must be numeric"

    if not (1e-6 <= lr <= 1.0):
        return None, "lr out of sane range [1e-6, 1.0]"
    if not (1 <= k <= 256):
        return None, "k out of sane range [1, 256]"

    hypothesis = str(raw.get("hypothesis", "")).strip() or "(no hypothesis given)"

    return {
        "model": model,
        "extra_fields": sorted(set(extra_fields)),
        "lr": round(lr, 7),
        "k": k,
        "seed": seed,
        "hypothesis": hypothesis,
    }, None


def call_llm_for_next_experiment(history):
    """Returns (proposal_dict_or_None, tokens_dict_or_None, error_str_or_None)."""
    try:
        from openai import OpenAI
    except ImportError:
        return None, None, "openai package not installed"

    try:
        client = OpenAI()
    except Exception as e:
        return None, None, f"could not construct OpenAI client: {e}"

    prompt = PLANNER_INSTRUCTIONS + "\n\nExperiment history so far:\n" + summarize_history(history)

    last_err = None
    for attempt in range(2):
        try:
            response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        except Exception as e:
            return None, None, f"OpenAI call failed: {e}"

        usage = getattr(response, "usage", None)
        tokens = {
            "input": getattr(usage, "input_tokens", None) if usage else None,
            "output": getattr(usage, "output_tokens", None) if usage else None,
        }

        raw = parse_json_object(getattr(response, "output_text", None))
        proposal, err = (None, "empty response") if raw is None else validate_proposal(raw)
        if proposal is not None:
            return proposal, tokens, None

        last_err = err
        prompt = PLANNER_INSTRUCTIONS + RETRY_SUFFIX + "\n\nExperiment history so far:\n" + summarize_history(history)

    return None, None, f"LLM did not return a valid proposal after retry ({last_err})"


# ==========================================
# HEURISTIC FALLBACK (used only if the LLM is unavailable / invalid / duplicate)
# ==========================================

def config_key(cfg):
    return (cfg["model"], tuple(sorted(cfg.get("extra_fields", []))),
            round(float(cfg["lr"]), 7), int(cfg["k"]), int(cfg.get("seed", 0)))


def heuristic_next_experiment(history, tried_keys):
    successes = [x for x in history if x.get("status") == "success"]

    if not any(x["model"] == "fm" for x in successes):
        return {"model": "fm", "extra_fields": [], "lr": 0.001, "k": 16, "seed": 0,
                "hypothesis": "Establish the FM pointwise baseline."}

    bpr_successes = [x for x in successes if x["model"] == "bpr"]
    if not bpr_successes:
        return {"model": "bpr", "extra_fields": [], "lr": 0.001, "k": 16, "seed": 0,
                "hypothesis": "Test whether pairwise BPR ranking improves over the pointwise FM baseline."}

    best_bpr = max(bpr_successes, key=lambda x: x["valid_primary"])
    for lr in (best_bpr["lr"] / 3, best_bpr["lr"] * 3):
        cfg = {"model": "bpr", "extra_fields": best_bpr.get("extra_fields", []),
               "lr": round(lr, 7), "k": best_bpr["k"], "seed": best_bpr.get("seed", 0),
               "hypothesis": f"Nearby learning rate to current best BPR lr={best_bpr['lr']}."}
        if config_key(cfg) not in tried_keys:
            return cfg

    return None


# ==========================================
# EXPERIMENT ENGINE
# ==========================================

def run_experiment_once(cfg):
    command = [
        sys.executable, cfg["script"],
        "--model", cfg["model"],
        "--lr", str(cfg["lr"]),
        "--k", str(cfg["k"]),
        "--seed", str(cfg["seed"]),
    ]
    if cfg.get("extra_fields"):
        command += ["--features", ",".join(cfg["extra_fields"])]

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=EXPERIMENT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, "timeout"

    if result.returncode != 0:
        print(result.stderr[-2000:])
        return None, "nonzero_exit"

    match = re.search(
        r"valid\s+GAUC\s+([0-9.]+)"
        r"\s+\|\s+nDCG@5\s+([0-9.]+)"
        r"\s+\|\s+primary\s+([0-9.]+)",
        result.stdout
    )
    if match is None:
        return None, "unparseable_output"

    return {
        "valid_gauc": float(match.group(1)),
        "valid_ndcg": float(match.group(2)),
        "valid_primary": float(match.group(3)),
    }, None


def run_experiment(cfg, retries=1):
    """Run with one bounded retry on failure (crash, timeout, unparseable output)
    before giving up on this config for the current iteration."""
    reason = None
    for attempt in range(retries + 1):
        result, reason = run_experiment_once(cfg)
        if result is not None:
            return result, None
        if attempt < retries:
            print(f"  attempt {attempt + 1} failed ({reason}); retrying...")
    return None, reason


def describe_config_diff(prev_cfg, new_cfg):
    """Human-readable diff of what actually changed between successive pipeline
    configurations (a config/parameter diff, not a literal source-file patch —
    the pipeline's behavior is driven entirely by these fields)."""
    if prev_cfg is None:
        return "(first experiment - no prior config to diff against)"
    lines = []
    for field in ("model", "extra_fields", "lr", "k", "seed"):
        old, new = prev_cfg.get(field), new_cfg.get(field)
        if old != new:
            lines.append(f"- {field}: {old}\n+ {field}: {new}")
    return "\n".join(lines) if lines else "(no change from previous config)"


def check_converged(successful_history, eps=CONVERGENCE_EPS, n=CONVERGENCE_N):
    """Converged when the running-best validation primary hasn't improved by
    more than eps over the last n successful iterations."""
    if len(successful_history) < n + 1:
        return False
    best_so_far = []
    best = -1.0
    for x in successful_history:
        best = max(best, x["valid_primary"])
        best_so_far.append(best)
    return (best_so_far[-1] - best_so_far[-1 - n]) <= eps


# ==========================================
# MAIN AUTONOMOUS LOOP
# ==========================================

def main():
    print("Agent starting...")
    history = load_history()
    run_start = time.time()
    total_tokens = {"input": 0, "output": 0}
    stop_reason = None
    last_cfg = history[-1] if history else None

    while True:
        if len(history) >= MAX_ITERATIONS:
            stop_reason = "iteration_cap"
            break
        if time.time() - run_start > WALL_CLOCK_LIMIT_S:
            stop_reason = "wall_clock_limit"
            break

        successes = [x for x in history if x.get("status") == "success"]
        if check_converged(successes):
            stop_reason = "converged"
            break

        tried_keys = {config_key(x) for x in successes}

        planner = "llm"
        proposal, tokens, err = call_llm_for_next_experiment(history)
        if tokens:
            total_tokens["input"] += tokens.get("input") or 0
            total_tokens["output"] += tokens.get("output") or 0

        if proposal is not None and config_key(proposal) in tried_keys:
            print(f"LLM proposed an already-tried config; falling back to heuristic this round.")
            proposal, err = None, "duplicate proposal"
        if proposal is None:
            print(f"LLM planner unavailable this round ({err}); using heuristic fallback.")
            proposal = heuristic_next_experiment(history, tried_keys)
            planner = "heuristic_fallback"

        if proposal is None:
            stop_reason = "no_new_candidates"
            break

        script = "bpr.py" if proposal["model"] == "bpr" else "baseline.py"
        cfg = dict(proposal, script=script)
        cfg["name"] = f"{cfg['model']}_{'+'.join(cfg['extra_fields']) or 'base'}_lr{cfg['lr']}_k{cfg['k']}_s{cfg['seed']}"

        print()
        print("=" * 50)
        print("Experiment", len(history) + 1, f"[{planner}]")
        print("Config:", {k: cfg[k] for k in ("model", "extra_fields", "lr", "k", "seed")})
        print("Hypothesis:", cfg["hypothesis"])
        diff = describe_config_diff(last_cfg, cfg)
        print("Diff vs previous:")
        print(diff)
        print("=" * 50)

        t0 = time.time()
        result, failure_reason = run_experiment(cfg)
        elapsed = time.time() - t0

        record = {
            "experiment_number": len(history) + 1,
            "name": cfg["name"],
            "script": cfg["script"],
            "model": cfg["model"],
            "extra_fields": cfg["extra_fields"],
            "lr": cfg["lr"],
            "k": cfg["k"],
            "seed": cfg["seed"],
            "planner": planner,
            "hypothesis": cfg["hypothesis"],
            "config_diff": diff,
            "wall_clock_s": round(elapsed, 1),
        }

        if result is None:
            record["status"] = "failed"
            record["failure_reason"] = failure_reason
            print(f"Experiment failed ({failure_reason}). Recorded and continuing...")
        else:
            record["status"] = "success"
            record.update({
                "valid_gauc": result["valid_gauc"],
                "valid_ndcg": result["valid_ndcg"],
                "valid_primary": result["valid_primary"],
            })
            print()
            print("Result: GAUC", result["valid_gauc"], "| nDCG@5", result["valid_ndcg"],
                  "| primary", result["valid_primary"])

        history.append(record)
        save_history(history)
        last_cfg = cfg

        successful = [x for x in history if x.get("status") == "success"]
        if successful:
            best = max(successful, key=lambda x: x["valid_primary"])
            print("Best so far: experiment", best["experiment_number"], "primary", best["valid_primary"])

    total_wall_clock = time.time() - run_start
    successful = [x for x in history if x.get("status") == "success"]
    best = max(successful, key=lambda x: x["valid_primary"]) if successful else None

    summary = {
        "stop_reason": stop_reason,
        "iterations_used": len(history),
        "iteration_cap": MAX_ITERATIONS,
        "wall_clock_limit_s": WALL_CLOCK_LIMIT_S,
        "total_wall_clock_s": round(total_wall_clock, 1),
        "convergence_rule": {"epsilon": CONVERGENCE_EPS, "N": CONVERGENCE_N},
        "llm_tokens": total_tokens,
        "successful": len(successful),
        "failed": len(history) - len(successful),
        "manual_interventions": 0,
        "best_experiment": best,
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=4)

    print()
    print("=" * 50)
    print("RESEARCH COMPLETE -", stop_reason)
    print("=" * 50)
    print("Experiments attempted:", len(history), "| successful:", len(successful),
          "| failed:", len(history) - len(successful))
    print("LLM tokens - input:", total_tokens["input"], "output:", total_tokens["output"])
    print("Total wall-clock: {:.1f}s".format(total_wall_clock))
    if best:
        print()
        print("Best experiment:", best["name"], "| primary:", best["valid_primary"])
    print(f"Summary written to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
