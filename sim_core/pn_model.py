from dataclasses import dataclass
from pm4py.objects.petri_net.obj import PetriNet, Marking


@dataclass
class PNModel:
    place_ids: list
    trans_ids: list
    labels: dict
    inputs: dict
    outputs: dict
    im: dict
    fm: dict


def place_id(p):
    return p.name


def trans_id(t):
    return t.name


def trans_label(t):
    if t.label is None:
        return ""
    s = str(t.label).strip()
    if s == "":
        return ""
    return s


def marking_dict(m):
    d = {}
    for p, c in m.items():
        d[place_id(p)] = int(c)
    return d


def build_empty_io(net):
    inputs = {}
    outputs = {}
    labels = {}

    for t in net.transitions:
        tid = trans_id(t)
        inputs[tid] = []
        outputs[tid] = []
        labels[tid] = trans_label(t)

    return inputs, outputs, labels


def fill_io_from_arcs(net, inputs, outputs):
    for a in net.arcs:
        s = a.source
        t = a.target

        if isinstance(s, PetriNet.Place) and isinstance(t, PetriNet.Transition):
            outputs_of_place_to_transition = inputs
            outputs_of_place_to_transition[trans_id(t)].append(place_id(s))

        elif isinstance(s, PetriNet.Transition) and isinstance(t, PetriNet.Place):
            outputs_of_transition_to_place = outputs
            outputs_of_transition_to_place[trans_id(s)].append(place_id(t))

        else:
            raise TypeError("Arc endpoints are not place transition or transition place")


def wrap_net(net, im, fm):
    place_ids = [place_id(p) for p in net.places]
    trans_ids = [trans_id(t) for t in net.transitions]

    inputs, outputs, labels = build_empty_io(net)
    fill_io_from_arcs(net, inputs, outputs)

    im_dict = marking_dict(im)
    fm_dict = marking_dict(fm)

    return PNModel(
        place_ids=place_ids,
        trans_ids=trans_ids,
        labels=labels,
        inputs=inputs,
        outputs=outputs,
        im=im_dict,
        fm=fm_dict,
    )


