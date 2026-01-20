import pm4py
import os
import sys

# A_Create Application
CREATE_APP_ID = "sid-CFB2384F-B409-401F-BEEB-248E78AA3C61"

def main():
    pnml_path = os.path.join("data", "process_model.pnml")
    print(f"Loading {pnml_path}...")
    net, im, fm = pm4py.read_pnml(pnml_path)

    print(f"Inspecting outputs of {CREATE_APP_ID} (A_Create Application)")
    
    target_trans = None
    for t in net.transitions:
        if t.name == CREATE_APP_ID:
            target_trans = t
            break
            
    if not target_trans:
        print("Not found.")
        return

    for arc in target_trans.out_arcs:
        target_place = arc.target
        print(f"  -> Place: {target_place.name}")
        
        # Check what follows this place
        print("     Outgoing Arcs from this place:")
        for out_arc in target_place.out_arcs:
            out_trans = out_arc.target
            print(f"       -> Transition: {out_trans.name} (Label: {out_trans.label})")

if __name__ == "__main__":
    main()
