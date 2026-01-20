import pandas as pd

from resources.ResourceManager import ResourceManager
from sim_core.engine import Engine
from sim_core.pn_model import wrap_net
from sim_core.bpmn_io import read_bpmn


def run_test():
    print("Preparing Dataframes...")
    permissions_path = ''
    availabilities_path = ''

    print("Loading BPMN Model...")
    bpmn_net, initial_marking, final_marking = read_bpmn('../data/process_model.bpmn')
    pn_model = wrap_net(bpmn_net, initial_marking, final_marking)

    print("Initializing Manager and Engine...")
    manager = ResourceManager()
    engine = Engine(pn_model, manager)

    print("Running Simulation...")
    engine.spawn()
    engine.run(max_events=10000)

    sim_log = pd.DataFrame(engine.log)
    print("\n--- Simulation Output ---")
    print(sim_log.head(10))

    sim_log.to_csv("test_output.csv", index=False)
    print("\nResults saved to 'test_output.csv'")


if __name__ == "__main__":
    run_test()
