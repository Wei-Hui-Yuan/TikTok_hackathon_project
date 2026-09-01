# TikTok_hackathon_project
# Autonomous ML Research Agent for KuaiRand-Pure Benchmark

This repository contains an end-to-end framework and an **Autonomous ML Research Agent** for short-video recommendation ranking on the KuaiRand-Pure benchmark. The agent systematically evaluates baseline models, performs feature engineering, and optimizes loss functions to maximize within-user recommendation metrics (GAUC and nDCG@5).

---

## Project Overview

The core objective is to autonomously iterate on an algorithmic pipeline using the KuaiRand-Pure training and validation sets to outperform the official baseline on a hidden test set.

* **Task**: Within-user ranking over logged impressions.


* **Label**: `long_view` (binary classification).


* **Primary Metric**: $\text{Primary} = \frac{\text{GAUC} + \text{nDCG@5}}{2}$.


* **Data Splits**:
* **Train**: April 8, 2022 – April 21, 2022 (1,141,112 rows).


* **Valid**: April 22, 2022 – April 28, 2022 (124,909 rows).


* **Test**: April 29, 2022 – May 8, 2022 (170,588 rows).





### Benchmark Scores Reference

| Model / Baseline | Valid GAUC | Valid nDCG@5 | Valid Primary | Test GAUC | Test nDCG@5 | Test Primary | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Random Baseline** | 0.4993 | 0.4675 | 0.4834 | 0.4996 | 0.4511 | 0.4753 | `baseline_scores.json`<br> |
| **Item Popularity** | 0.6387 | 0.5227 | 0.5807 | 0.6308 | 0.5121 | 0.5715 | `baseline_scores.json`<br> |
| **Official FM Baseline** | **0.6674** | **0.5357** | **0.6016** | **0.6610** | **0.5282** | **0.5946** | `baseline_scores.json`<br> |
| **Oracle Ceiling** | 1.0000 | 0.6968 | 0.8484 | 1.0000 | 0.7289 | 0.8645 | `baseline_scores.json`<br> |

---

## Repository Structure

* `data.py`: Loads date-based data splits and handles categorical feature discretization/bucketization.


* `evaluate.py`: Implements official evaluation metrics (Mann-Whitney U for GAUC and nDCG@5).


* `baseline.py`: Contains implementations for Random, Item Popularity, and standard Factorization Machine (FM) models.


* `bpr.py`: Extends FM with pairwise Bayesian Personalized Ranking (BPR) loss optimization.


* `ablation_features.py`: Runs feature expansion ablations comparing standard 5-field features against 9-field and 13-field variants.


* `agent.py`: Executes the autonomous research agent loop. Each iteration, an LLM (OpenAI API) is shown the full experiment history and proposes the next experiment — model/loss, feature set, and hyperparameters — with a stated hypothesis; falls back to a deterministic heuristic if the API is unavailable or proposes an invalid/duplicate config.


* `submit.py`: Validates submission alignment and exports prediction CSV files.


* `test_openai.py`: Verification script for LLM API connection integration.


* `baseline_scores.json`: Benchmark target metrics and dataset statistics.



---

## Setup and Installation

### Prerequisites

* Python 3.8+
* Standard libraries + `numpy` (no complex framework dependencies required for pure baselines)



```bash
# Clone the repository
git clone https://github.com/your-username/TikTok_hackathon_project.git
cd TikTok_hackathon_project

# Install required packages
pip install numpy openai

# Required for the autonomous agent's LLM-driven planner (agent.py)
export OPENAI_API_KEY=your-key-here
# Optional: override the default model (see test_openai.py)
export AGENT_OPENAI_MODEL=gpt-5.6

```

### Data Placement

Ensure the KuaiRand-Pure dataset files are unzipped and placed in `./KuaiRand-Pure/data`:

* `log_standard_4_08_to_4_21_pure.csv`

* `log_standard_4_22_to_5_08_pure.csv`

* `video_features_basic_pure.csv`

* `user_features_pure.csv`


---

## Steps to Reproduce Results

### 1. Reproduce Official Baseline (FM)

Run the official single-CPU Factorization Machine baseline:

```bash
python3 baseline.py --model fm --lr 0.001 --k 16 --seed 0

```

### 2. Run Pairwise BPR Training

Train the model using Bayesian Personalized Ranking loss:

```bash
python3 bpr.py --model bpr --lr 0.001 --k 16 --seed 0

```

### 3. Run Feature Ablation Experiments

Evaluate the impact of extending the feature space from 5 basic fields to 9 (item side) and 13 (CWM full) fields:

```bash
python3 ablation_features.py ./KuaiRand-Pure/data

```

### 4. Execute the Autonomous Research Agent

Launch the automated agent loop (requires `OPENAI_API_KEY`, see Setup above). Each iteration, an LLM proposes the next experiment across model/loss, feature set, and hyperparameters with a stated hypothesis; the agent validates the proposal, runs it, and logs the result to `experiment_log.json`. The run stops on the challenge's convergence rule (validation primary improves by ≤ 0.002 over 3 consecutive iterations), a 50-iteration cap, or a 6-hour wall-clock ceiling — whichever comes first. A run-level summary (iterations used, LLM token usage, wall-clock, best result) is written to `experiment_summary.json`:

```bash
python3 agent.py

```

### 5. Generate and Check Submission

Create a valid submission CSV file and check its alignment:

```bash
# Generate submission using official FM baseline
python3 submit.py submission.csv --make --split test

# Validate submission formatting and alignment
python3 submit.py submission.csv --check --split test

# Score submission locally on validation split
python3 submit.py submission.csv --score --split valid

```

---

## Autonomous Agent Architecture

Each iteration of `agent.py` runs the following loop:

1. **Propose**: The LLM is shown the full experiment history — every prior config, its stated hypothesis, and its result or failure reason — and asked to propose the next experiment across three real pipeline levers: **model/loss** (`fm` pointwise binary cross-entropy vs. `bpr` pairwise Bayesian Personalized Ranking), **feature set** (any combination of 8 optional item-side/user-side fields on top of the fixed 5-field base), and **hyperparameters** (`lr`, `k`, `seed`), together with a one- or two-sentence hypothesis for why.


2. **Validate**: The proposal is whitelist-validated before it is ever used to build a command — model/feature names are checked against known choices, `lr`/`k` are range-checked. If the LLM is unavailable, returns invalid JSON after one corrective retry, or proposes a configuration already tried, the agent falls back to a deterministic heuristic (start with the FM baseline, then BPR, then sweep nearby learning rates) so a single external-dependency failure can't stall the run. Each logged record notes which path produced it (`"planner": "llm"` vs. `"heuristic_fallback"`).


3. **Execute**: The chosen config is run as a subprocess (`baseline.py` or `bpr.py`) with a wall-clock timeout and one automatic retry on failure (crash, timeout, or unparseable output), so isolated failures are recovered from rather than crashing the run.


4. **Record**: Hypothesis, the resulting config diff versus the previous experiment, validation GAUC/nDCG@5/primary (or the failure reason), and per-experiment wall-clock are persisted to `experiment_log.json`.


5. **Check convergence**: The run stops once validation-best primary hasn't improved by more than `ε = 0.002` over the last `N = 3` successful iterations, or once the 50-iteration/6-hour budget is exhausted — whichever comes first. A final `experiment_summary.json` records the stop reason, iterations used, total LLM token usage, total wall-clock, and the best experiment.


Note on scope: the LLM chooses *from* a validated, whitelisted parameter space (model, feature columns, hyperparameters) rather than generating and executing arbitrary Python each round — an unattended loop that `exec`s LLM-authored code was judged too risky for this timeline. The "config diff" logged per iteration reflects genuine pipeline-behavior changes (a different loss function, a different feature set, different hyperparameters), just expressed as a validated config rather than a literal source-file patch.



---

## Limitations and Future Work

### Limitations

* **Pure NumPy Framework**: The current codebase relies on a custom NumPy engine. While lightweight, it limits scaling to complex neural architectures like Deep Interest Networks (DIN) or DLRM.


* **Constrained Decision Space**: `agent.py`'s LLM planner chooses from a validated set of levers (model/loss, feature columns, hyperparameters) rather than generating and executing arbitrary new code each iteration; it cannot yet invent a new loss function or model architecture outside what's already implemented in `baseline.py`/`bpr.py`.


* **Pairwise Sampling Cost**: BPR pair generation is performed in CPU memory, restricting real-time online sampling flexibility.



### Future Improvements

* **Deep Learning Integration**: Migrate backend training to PyTorch/RecBole to support multi-layer perceptrons, cross networks, and attention mechanisms.
* **Freeform Code-Generation Agent**: Extend the LLM planner from choosing among validated parameters to actually writing and applying new model/loss code each iteration (in the style of AIDE/MLE-Bench-style agents), with sandboxed execution and automated correctness checks before a generated change is run.


* **Multi-Task Learning**: Extend loss functions to joint click-through rate (CTR) and watch-time regression targets.

---

