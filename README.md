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
├── data/                       # Data folder containing bpmn model and event log
├── decision_analysis/
├── processing_times/
    ├──Advanced_Model
    ├──Basic_Models
├── resources/
    ├── allocation/
    ├── availabilities/
    ├── permissions/
├── sim_core/
├── sim_output/
├── simulation_evaluation/
├── spawn_rates/
    ├── artifacts
├── .gitignore
├── README.md
├── requirements.txt
├── run_simulation.py            # Main execution script
└── setup.sh      
                     
     
```

## How the Simulator Works

Our simulator is a **data-driven discrete-event simulation** built around a global clock and a **heapq-based event queue**. The underlying BPMN model is transformed into an executable **Petri net**. Events are scheduled as **SPAWN**, **START**, **COMPLETE**, and **RETRY**. The architecture is splitted into spawn rates, routing, resources, and processing times.



## Notebooks
- **Decision Analysis:**
    - `decision_analysis/xor_identification.ipynb` – XOR discovery  
    - `decision_analysis/probabilities.ipynb` – branching probabilities → `decision_prob_rules.json`  
- **Processing Times:** `processing_times/Processing_Times.ipynb` – processing-time models (basic vs quantile regression), look up the information sheet for further information
- **Resource Management:**
    - `resources/availabilities/resource_availabilities.ipynb` – Extracts availabilities schedule
    - `resources/permissions/resource_permissions.ipynb` - Performs clustering for permission model
---
