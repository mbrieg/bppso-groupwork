# Group Exercise - Business Process Prediction, Simulation, and Optimization
## Business Process Simulation Model

## Description

## Getting started
This project requires a Python environment (preferably Conda) and the following external libraries with versions listed in `requirements.txt`.
### Dependencies
- Python 3.12.12
- Core libraries: pm4py, pandas, scikit-learn

### Installing
To set up the environment, clone the repository and use `pip` to install all dependencies:
```
pip install -r requirements.txt
```

### Executing program
1. Clone the repo
    ```
    git clone https://github.com/mbrieg/bppso-groupwork.git
    ```
- **Start the Jupyter Server:**
    ```
    jupyter notebook .
    ```

- **Open and Run:** Execute 'run_simulation.py'

## Project Structure
```
.
├── data/                       # Data folder containing bpmn model and event log
├── decision_analysis/ 
    
├── processing_times/
├── resources/
    ├── availabilities/
    ├── permissions/
├── sim_core/
├── sim_output/
├── test/
├── README.md
├── requirements.txt
├── run_simulation.py       
└── setup.sh      
```