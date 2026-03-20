# Business Process Prediction, Simulation, and Optimization
## Group Exercise – Business Process Simulation Model (BPIC 2017)

This repository contains a **discrete event simulation** of the BPIC 2017 loan application process. 
The simulator spawns new cases, enforces the **control-flow**  defined by the BPMN model, handles **resource availability, processing times & permissions**, and produces a **simulated event log**.

### Dataset
Based on the BPIC 2017 Application log (4TU.ResearchData):

DOI: **10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b**

---

## Quick Start

### 1) Setup Environment
    ```
    git clone https://github.com/mbrieg/bppso-groupwork.git
    cd bppso-groupwork
    pip install -r requirements.txt
    ```

### 2) Run Simulation
```
python run_simulation.py
```
### 3) Output Directory
```
sim_output/final_simulation_log.csv
```

### 4) Repository Structure
```
.
├── data/                           # BPMN model and raw event log
├── decision_analysis/
│   ├── DPManager.py                # Engine interface – selects router per XOR gateway
│   ├── BasicRouter.py              # Probability-based routing
│   ├── AdvancedRouter.py           # C4.5 decision-tree routing
│   ├── c45_tree.py                 # C4.5 implementation
│   ├── decision_prob_rules.json    # Branching probabilities (BasicRouter input)
│   └── simulation_brain.pkl        # Trained C4.5 models (AdvancedRouter input)
├── processing_times/
│   ├── Advanced_Model/
│   └── Basic_Models/
├── resources/
│   ├── availabilities/
│   └── permissions/
├── sim_core/
├── sim_output/
├── simulation_evaluation/
│   ├── evaluation.ipynb            # Main evaluation (metrics 1–4 vs real log)
│   ├── firing_decision.ipynb       # Workforce optimization experiment
│   ├── 9to5.ipynb                  # Alternative 9-to-5 schedule experiment
│   └── resource_allocation/
│       ├── run_allocation_study.py # Runs 6 allocation methods × 5 replications
│       └── resource_allocation_evaluation.ipynb
├── spawn_rates/
│   └── artifacts/
├── test/
├── README.md
├── requirements.txt
├── run_simulation.py
└── setup.sh
```

## How the Simulator Works

Our simulator is a **data-driven discrete-event simulation** built around a global clock and a **heapq-based event queue**. The underlying BPMN model is transformed into an executable **Petri net**. Events are scheduled as **SPAWN**, **START**, **COMPLETE**, and **RETRY**. The architecture is splitted into spawn rates, routing, resources, and processing times.

## Decision Analysis

`DPManager` routes cases through XOR gateways in three modes:

| Mode | Router | Description |
|---|---|---|
| `random` | — | Uniform random over enabled transitions |
| `basic` | `BasicRouter` | Probability-based, learned from the real log |
| `advanced` | `AdvancedRouter` | C4.5 decision tree usage |

Per-case dynamic features (offer count, rejection/acceptance history) are tracked live during simulation and injected into the C4.5 tree at each decision point.

## Simulation Evaluation

### Resource Allocation Study

`run_allocation_study.py` runs 6 allocation methods × 5 replications (14 simulated days each) and saves results to `resource_allocation/data/<method>/run_NN.csv`.

| Method | δ |
|---|---|
| `random`, `round_robin`, `shortest_queue`, `batch_k5` | — |
| `advanced_local` | 5 |
| `advanced_global` | 10 |

`resource_allocation_evaluation.ipynb` compares all methods across cycle time, occupation, fairness MAD, OLI, and CHR.

### Main Evaluation (`evaluation.ipynb`)

| # | Metric | Direction |
|---|---|---|
| 1 | Average Cycle Time | lower  |
| 2 | Resource Occupation | higher  |
| 3 | Resource Fairness (MAD) | lower  |
| 4 | Outcome Distribution Fidelity (1 − JSD) | higher  |

### Workforce Optimization (`firing_decision.ipynb`)

Identifies the least-utilized employees from the best run, removes them from the schedule, reruns the simulation, and measures the impact on basic metrics.

### Alternative Schedule Experiment (`9to5.ipynb`)

Generates a uniform 9-to-5 availability schedule and compares metrics against the baseline advanced simulation.


## Notebooks

**Decision Analysis**
- `decision_analysis/xor_identification.ipynb` – XOR gateway discovery & C4.5 model training -> `simulation_brain.pkl`
- `decision_analysis/probabilities.ipynb` – branching probability estimation -> `decision_prob_rules.json`
- `decision_analysis/decision_evaluation.ipynb` – JSD-based comparison: random vs basic vs advanced routing
- `decision_analysis/case_attributes_distributions.ipynb` – case attribute distribution analysis for AdvancedSpawner

**Processing Times**
- `processing_times/Processing_Times.ipynb` – processing-time models (basic vs quantile regression)

**Simulation Evaluation**
- `simulation_evaluation/resource_allocation/resource_allocation_evaluation.ipynb` – allocation method comparison
- `simulation_evaluation/evaluation.ipynb` – main evaluation vs real log
- `simulation_evaluation/firing_decision.ipynb` – workforce optimization
- `simulation_evaluation/9to5.ipynb` – 9-to-5 schedule experiment

