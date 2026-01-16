
import pm4py
import os
import pandas as pd
import random

from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.objects.log.obj import EventLog

class ProcessDiscovery:
    """
    From the event log we discover using Inductive miner a bpmn model.
    """
    def __init__(self, dependency_threshold=0.5, and_threshold=0.65, loop_two_threshold=0.5):
        #self.dep_thresh = dependency_threshold
        #self.and_thresh = and_threshold
        #self.loop_two_thresh = loop_two_threshold
        self.noise_threshold =0.2

    def discover(self, log):
        #print(f"Mining process model (Dep: {self.dep_thresh}, and: {self.and_thresh})...")
        print(f"Mining process model by inductive miner (Dep: {self.noise_threshold})...")
        net, im, fm = pm4py.discover_petri_net_inductive(
            log,
            #dependency_threshold=self.dep_thresh,
            #and_threshold=self.and_thresh,
            #loop_two_threshold=self.loop_two_thresh
            noise_threshold=self.noise_threshold
        )
        return net, im, fm

    def evaluate_model(self, log, net, im, fm, sample_size= 1000):
        """
        Fitness, Precision, Generalization, Size, Density like in the first exercise.
        """
        print("Calculating metrics...")
        metrics = {}
        log = pm4py.convert_to_event_log(log)
        log_list = list(log)
        real_size = len(log_list)

        if real_size > sample_size:
            print(f"  Log size ({real_size}) is large. Sampling {sample_size} traces...")
            
            sampled_data = random.sample(log_list, sample_size)
            sampled_log = EventLog(sampled_data)
        else:
            print(f"  Log size ({real_size}) is small enough. Using full log.")
            sampled_log = EventLog(log_list)

        # 1. FITNESS (Token-Based)
        fitness = replay_fitness.apply(sampled_log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)
        metrics['Fitness (Token-Based)'] = fitness['log_fitness']

        # 2. PRECISION
        prec = precision_evaluator.apply(sampled_log, net, im, fm)
        metrics['Precision'] = prec

        # 3. GENERALIZATION
        gen = generalization_evaluator.apply(sampled_log, net, im, fm)
        metrics['Generalization'] = gen

        n_places = len(net.places)
        n_transitions = len(net.transitions)
        n_arcs = len(net.arcs)
        n_nodes = n_places + n_transitions

        # 4. SIZE
        metrics['Size (Nodes)'] = n_nodes
        metrics['Size (Arcs)'] = n_arcs
        metrics['Total Elements'] = n_nodes + n_arcs

        # 5. DENSITY
        max_arcs = 2 * n_places * n_transitions
        metrics['Density'] = n_arcs / max_arcs if max_arcs > 0 else 0.0

        # 6. CONNECTIVITY
        metrics['Avg Connectivity (Degree)'] = n_arcs / n_nodes if n_nodes > 0 else 0.0

        return metrics
    
def main():
    log_path = "/Users/zeynepcetin/bppso-groupwork-1/data/BPI Challenge 2017.xes.gz"

    if os.path.exists(log_path):
        print("Loading Log...")
        log = pm4py.read_xes(log_path)
    
        miner = ProcessDiscovery(
            dependency_threshold=0.8,
            and_threshold=0.65
        )
    
        net, im, fm = miner.discover(log)
    
        results = miner.evaluate_model(log, net, im, fm, sample_size=1000)
    
        print("\nModel Evaluation Results")
        df_results = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
        print(df_results)
    
        print("\nVisualizing Model:")
        pm4py.view_petri_net(net, im, fm)
    
    else:
        print(f"Error: File not found -> {log_path}")

    print("\n Saving..")
    output_pnml_path = "discovered_model.pnml"

    pm4py.write_pnml(net,im,fm,output_pnml_path)
    print(f" Petri Net saved to: {os.path.abspath(output_pnml_path)}")


if __name__ == "__main__":
    main()
