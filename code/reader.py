import onnx
import onnx.numpy_helper
from onnx import numpy_helper
import numpy as np
import networkx as nx

import utils



def take_first(elem):    
    return elem[0] 
def take_sec(elem):
    return elem[1] 


def network_data(onnx_filename):
    """
    Given the name of the onnx example file, load it and convert it into a
    NetworkX graph. Returns this NetworkX.

    Return:
    NetworkX directed graph of the loaded onnx file.
    """
    onnx_weights = {}
    onnx_model = onnx.load(onnx_filename)
    model_data  = onnx_model.graph.initializer
    # Get the weights and biases out
    for init in model_data:
        weight_bias = numpy_helper.to_array(init)
        onnx_weights[init.name] = weight_bias
    
    keys = list(onnx_weights.keys())
    inp_size = len(onnx_weights[keys[0]][0])

    #Get the layer sizes
    layer_sizes = []
    for key in keys:
        size = len(onnx_weights[key])
        if(key.split(".")[1]=="bias"):
            layer_sizes.append(size)
    G = nx.DiGraph()

    #Construct the input layer
    for i in range(inp_size):
        G.add_node((0,i))
    
    #Construct the rest layers
    
    layer_id=0
    for i in range(len(layer_sizes)):
        sizes=layer_sizes[i]
        layer_id+=1
        for i in range(sizes):
            G.add_node((layer_id,i))

    w1=onnx_weights["first.weight"]
    b1=onnx_weights["first.bias"]
    w2=onnx_weights["second.weight"]
    b2=onnx_weights["second.bias"]

    layer_id = 0

    for i in range(len(w1[0])):
        for j in range(len(w1)):
            G.add_edges_from([((layer_id,i),(layer_id+1,j),{"weight":w1[j][i]})])
    
    layer_id= layer_id+1
    
    for i in range(len(b1)):
        G.nodes[(layer_id,i)]["bias"]=b1[i]
        
    for i in range(len(w2[0])):
        for j in range(len(w2)):
            G.add_edges_from([((layer_id,i),(layer_id+1,j),{"weight":w2[j][i]})])

    layer_id = layer_id+1

    for i in range(len(b2)):
        G.nodes[(layer_id,i)]["bias"]=b2[i]

    return G



def get_property(fname, n_out_vars):
    """
    Opens the .prop file located at fname

    Arguments:
    fname       -   The name of the .prop file to load
    n_out_vars  -   The number of output variables

    Returns:
    1.  The linear equations representing the output side property.
    2.  A list of (lb,ub) pairs for each input
    """
    f = open(fname)
    dict_in_op_prop = eval(f.read())
    inputs = dict_in_op_prop['input']
    outputs= dict_in_op_prop['output']
    sorted(inputs,key=take_first)
    intervals = []
    for i in inputs:
        inter = i[1]
        if 'Lower' not in inter.keys():
            l=  float('-inf')
        else:
            l = inter['Lower']
        if 'Upper' not in inter.keys():
            u =  float('inf')
        else:
            u = inter['Upper']
        
        intervals.append((l,u))
    
    coeff_in_eq = []
    for i in outputs:
        temp = i[0]
        for kind, bound in i[1].items():
            bound = bound if kind == 'Upper' else -bound
            coeffiecents = [0]*(n_out_vars+1)
            flag = True if kind == 'Upper' else False
            for k in temp: 
                if flag :
                    coeffiecents[k[1]] = k[0]    
                else :
                    coeffiecents[k[1]] = -1*k[0]
            coeffiecents.append(bound)
            coeff_in_eq.append(coeffiecents)

    #if config.DEBUG:
    #    utils.log(coeff_in_eq)
    #    utils.log(intervals)

    return coeff_in_eq, intervals


    
if __name__ == "__main__":


    import sys
    test_no = int(sys.argv[1])

    if test_no == 0:
        """
        Test loading onnx
        """
    
        G = network_data( 
                "../networks/path_to_save_model.onnx" )

        print( "Layer sizes: ", list( map( len, utils.getLayers( G ))))
