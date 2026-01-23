# Business Process Prediction, Simulation, and Optimization  
## Group Exercise – Business Process Simulation Model (BPIC 2017)

This repository contains a **event simulation** of the BPIC 2017 *Application* process. 
The simulator spawns new cases, enforces the **control-flow**  defined by the BPMN model, handles **resource availability, processing times & permissions**, and produces a **simulated event log**.

### Dataset
Based on the BPIC 2017 Application log (4TU.ResearchData):

**10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b**

---

## Quick Start

### 1) Setup environment
1. Clone the repo
    ```
    git clone https://github.com/mbrieg/bppso-groupwork.git
    ```
- **Start the Jupyter Server:**
    ```
    jupyter notebook .
    ```
- **install dependencies**
    ```
    pip install -r requirements.txt
    ```



### 2) Run Simulation
```
python run_simulation.py
```
### 3) Output
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
    ├── availabilities/
    ├── permissions/
├── sim_core/
├── sim_output/
├── spawn_rates/
    ├── artifacts
├── test/
├── README.md
├── requirements.txt
├── run_simulation.py       
└── setup.sh      
                     
     
```

## How the Simulator Works

Our simulator is an **Event Simulation** with a global clock and a **heapq based event queue**. The BPMN model is transformed into an executable **Petri net**. Events are scheduled as **SPAWN**, **START**, **COMPLETE**, and **RETRY**. The architecture is modular: spawn rates, routing, resources, and timing.



## Notebooks
- `decision_analysis/xor_identification.ipynb` – XOR discovery  
- `decision_analysis/probabilities.ipynb` – branching probabilities → `decision_prob_rules.json`  
- `processing_times/Processing_Times.ipynb` – processing-time models (basic vs quantile regression), look up the information sheet for further information

---
