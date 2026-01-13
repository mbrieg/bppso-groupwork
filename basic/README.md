**1.4 Data-Driven Decision Point Analysis BPIC 2017**

This module implements a Context/History-Aware Branching Strategy for loan application process simulation. The main idea is to calculate the probabilities based on the case history (preceding activity), instead of using random routing at XOR gateways.

The Approach : **Conditioned Probability ($P(Next \mid Current, Previous)$)**.
The model learns from the event log that, e.g., the cases coming from the *Assess Potential Fraud* have a significantly higher probability of being *Denied* than cases comign from *Complete Application*

**basic_decision_point_analysis.py**:
-Logic for parsing the Petri net
-Training the probabilistic model
-Routing function used by the core simulator

**test_basic_analysis.py**:
-Runs stability checks:
    -Horizon Sensitivty
    -Predictive Accuracy Tests (Log-Loss)
    -Raw Count Verifications


**Logic:** 
**1. Backtracking:** Petri nets often contain invisible (tau) transitions that obscure the real process. 
A decision point (XOR) might therotically be preceded by an invisible transition, which actually gives us no business context.
In order to overcome this possible problem, **get_preset_labels_with_backtracking** uses BFS to bactrack through tau transitions until it finds a business activity to use as a **trigger**.

**2. Conditioned First-Hit Counting:** Instead of simply counting arcs, we analyse the causal dependencies in the log: 
**Trigger**: We detect when a case enters a deciison point
**Horizon (60)**: We scan forward up to 60 steps. If a valid outcome is found, we record the link. If not, the episode is evaluated as noise. 

**3. Hiearchical Routing**: **route_at_decision_point** function uses a 3 tier fallback strategy:
**1. Context-Aware**: If we've seen the specific history (Transition(prev) --> Place --> Transition(next)), use the specific prob. 
**2. Marginal Fallback**: If the history is new, use the global avg. probability for that place.
**3. Random Fallback**: If no data exists, select randomly amonged enabled transitions.

**test_basic_analysis.py**
The model is consistenly tested using test_basic_analysis.py in order to create a robust decision point analysis.
**Predictive Power**: Comparing the condition-based model to the baseline model
The log-loss improvement ca. 0,10 shows that knowing the preceding activity significantly improves prediction accuracy.
**Stability Analysis**: Different size of horizons are tested (20,50,80,120)
H20 --> unstable High L1 distance
H60 --> L1 distance < 0.01
Horizon=60 is selected for the final simulation.
**Realism**: 
The model's predicted probabilities match the raw event log counts