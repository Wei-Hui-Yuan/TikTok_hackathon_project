import subprocess
import re
import json

MAX_EXPERIMENTS = 8
LOG_FILE = "experiment_log.json"


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
# RESEARCH BRAIN
# ==========================================

def choose_next_experiment(history):

    # --------------------------------------
    # Step 1: Establish the FM baseline
    # --------------------------------------

    if len(history) == 0:
        return {
            "name": "fm_baseline",
            "script": "baseline.py",
            "model": "fm",
            "lr": 0.001,
            "k": 16,
            "seed": 0,
            "reason": "Establish the FM validation baseline."
        }

    # --------------------------------------
    # Step 2: Try our starting BPR point
    # --------------------------------------

    bpr_experiments = [
        x for x in history
        if x.get("model") == "bpr"
        and x.get("status") == "success"
    ]

    if len(bpr_experiments) == 0:
        return {
            "name": "bpr_lr_0.001",
            "script": "bpr.py",
            "model": "bpr",
            "lr": 0.001,
            "k": 16,
            "seed": 0,
            "reason": (
                "Test whether pairwise BPR ranking improves "
                "over the pointwise FM baseline."
            )
        }


    # --------------------------------------
    # Step 3: Find best BPR learning rate
    # --------------------------------------

    best_bpr = max(
        bpr_experiments,
        key=lambda x: x["valid_primary"]
    )

    best_lr = best_bpr["lr"]


    # --------------------------------------
    # Step 4: Generate nearby candidates
    # --------------------------------------

    candidates = [
        best_lr / 3,
        best_lr * 3
    ]


    # --------------------------------------
    # Step 5: Don't repeat experiments
    # --------------------------------------

    tried_lrs = [
        x.get("lr")
        for x in bpr_experiments
    ]

    for lr in candidates:

        # Round so floating point doesn't create
        # silly names like 0.0003333333333333333
        lr = round(lr, 7)

        if lr not in tried_lrs:
            return {
                "name": f"bpr_lr_{lr}",
                "script": "bpr.py",
                "model": "bpr",
                "lr": lr,
                "k": 16,
                "seed": 0,
                "reason": (
                    f"The current best BPR learning rate is {best_lr}. "
                    f"Test nearby learning rate {lr}."
                )
            }


    # --------------------------------------
    # Nothing new found
    # --------------------------------------

    return None


# ==========================================
# EXPERIMENT ENGINE
# ==========================================

def run_experiment(experiment):

    print()
    print("Running experiment:", experiment["name"])

    command = [
        "python3",
        experiment["script"],
        "--model",
        experiment["model"]
    ]

    if "lr" in experiment:
        command.extend([
            "--lr",
            str(experiment["lr"])
        ])

    if "k" in experiment:
        command.extend([
            "--k",
            str(experiment["k"])
        ])

    if "seed" in experiment:
        command.extend([
            "--seed",
            str(experiment["seed"])
        ])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print("Experiment failed!")
        print(result.stderr)

        return None

    match = re.search(
        r"valid\s+GAUC\s+([0-9.]+)"
        r"\s+\|\s+nDCG@5\s+([0-9.]+)"
        r"\s+\|\s+primary\s+([0-9.]+)",
        result.stdout
    )

    if match is None:

        print("Could not find validation score!")

        return None

    return {
        "valid_gauc": float(match.group(1)),
        "valid_ndcg": float(match.group(2)),
        "valid_primary": float(match.group(3))
    }


# ==========================================
# MAIN AUTONOMOUS LOOP
# ==========================================

print("Agent starting...")

history = load_history()

while True:

    # --------------------------------------
    # 1. Decide what to try next
    # --------------------------------------
    if len(history) >= MAX_EXPERIMENTS:
        print()
        print("Experiment budget exhausted.")
        break

    experiment = choose_next_experiment(history)

    if experiment is None:
        print()
        print("No new experiments available.")
        break

    print()
    print("=" * 50)
    print("Experiment", len(history) + 1)
    print("Agent chose:", experiment["name"])
    print("Reason:", experiment["reason"])
    print("=" * 50)

    # --------------------------------------
    # 2. Run experiment
    # --------------------------------------

    result = run_experiment(experiment)

    # --------------------------------------
    # 3. Handle failure
    # --------------------------------------

    if result is None:

        print("Experiment failed.")

        record = {
            "experiment_number": len(history) + 1,
            "name": experiment["name"],
            "script": experiment["script"],
            "model": experiment["model"],

            "lr": experiment.get("lr"),
            "k": experiment.get("k"),
            "seed": experiment.get("seed"),

            "reason": experiment.get("reason"),

            "status": "success",

            "valid_gauc": result["valid_gauc"],
            "valid_ndcg": result["valid_ndcg"],
            "valid_primary": result["valid_primary"]
        }

        history.append(record)
        save_history(history)

        print("Failure recorded. Continuing...")
        continue

    # --------------------------------------
    # 4. Record successful experiment
    # --------------------------------------

    record = {
        "experiment_number": len(history) + 1,

        "name": experiment["name"],
        "script": experiment["script"],
        "model": experiment["model"],

        "lr": experiment.get("lr"),
        "k": experiment.get("k"),
        "seed": experiment.get("seed"),

        "status": "success",

        "valid_gauc": result["valid_gauc"],
        "valid_ndcg": result["valid_ndcg"],
        "valid_primary": result["valid_primary"]
    }

    history.append(record)
    save_history(history)

    print()
    print("Result:")
    print("GAUC:", result["valid_gauc"])
    print("nDCG@5:", result["valid_ndcg"])
    print("Primary:", result["valid_primary"])

    # --------------------------------------
    # 5. Find current best
    # --------------------------------------

    successful = [
        x for x in history
        if x.get("status") == "success"
    ]

    best_experiment = max(
        successful,
        key=lambda x: x["valid_primary"]
    )

    print()
    print("Best experiment so far:")
    print("Experiment:", best_experiment["experiment_number"])
    print("Name:", best_experiment["name"])
    print("Primary:", best_experiment["valid_primary"])


# ==========================================
# FINAL REPORT
# ==========================================

successful = [
    x for x in history
    if x.get("status") == "success"
]

if successful:

    best_experiment = max(
        successful,
        key=lambda x: x["valid_primary"]
    )

    print()
    print("=" * 50)
    print("RESEARCH COMPLETE")
    print("=" * 50)

    print("Experiments attempted:", len(history))

    print(
        "Successful:",
        len(successful)
    )

    print(
        "Failed:",
        len(history) - len(successful)
    )

    print()
    print("Best experiment:")
    print("Name:", best_experiment["name"])
    print("Primary:", best_experiment["valid_primary"])