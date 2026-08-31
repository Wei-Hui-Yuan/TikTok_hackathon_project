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


* `agent.py`: Executes the autonomous research agent loop to sweep hyperparameters and optimize validation performance.


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

Launch the automated agent loop. The agent executes up to 8 automated iterations, adjusting learning rates and logging output to `experiment_log.json`:

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

The agent operates through a multi-step execution loop:

1. **Baseline Verification**: Establishes initial baseline validation performance.


2. **Pairwise Paradigm Shift**: Switches from pointwise binary cross-entropy to pairwise BPR loss.


3. **Adaptive Hyperparameter Tuning**: Automatically adjusts learning rates ($\text{lr} \times 3$, $\text{lr} / 3$) based on validation primary score feedback.


4. **State Persistence**: Records hypotheses, configuration parameters, and execution outcomes directly into `experiment_log.json`.



---

## Limitations and Future Work

### Limitations

* **Pure NumPy Framework**: The current codebase relies on a custom NumPy engine. While lightweight, it limits scaling to complex neural architectures like Deep Interest Networks (DIN) or DLRM.


* **Heuristic Search Space**: The existing `agent.py` uses heuristic candidate generation rather than full LLM-driven code modification.


* **Pairwise Sampling Cost**: BPR pair generation is performed in CPU memory, restricting real-time online sampling flexibility.



### Future Improvements

* **Deep Learning Integration**: Migrate backend training to PyTorch/RecBole to support multi-layer perceptrons, cross networks, and attention mechanisms.
* **LLM-Driven Agent Brain**: Replace rule-based logic in `agent.py` with full LLM code-generation agents (using OpenAI or Trae APIs) to autonomously edit features and architecture files.


* **Multi-Task Learning**: Extend loss functions to joint click-through rate (CTR) and watch-time regression targets.

---

