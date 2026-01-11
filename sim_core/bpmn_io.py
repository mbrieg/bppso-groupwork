from pm4py.objects.bpmn.importer import importer as importer
from pm4py.objects.conversion.bpmn import converter as converter

def read_bpmn(bpmn_path):
    bpmn_graph = importer.apply(bpmn_path)
    net, im, fm = converter.apply(bpmn_graph)
    return net, im, fm
