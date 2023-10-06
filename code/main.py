import contrastive
import property_encode
import reader
import ast 
import copy
import utils
import property_encode
import numpy as np
from PIL import Image, ImageDraw
import torch

def compute_gradient( G, inp , end_relu):
    """
    Evaluates the network on the given input `inp`. Input can also be a
    stack of vectors.
    
    Returns:
    
    1.  The vector of return values
    2.  A list with the values at each layer. At each layer, if the layer
        has a ReLU, the value is the output of the ReLU, or if the layer is
        an input layer, the value is the input, and otherwise the value is
        the output of the linear layer.
    """
    layers = utils.getLayers(G)
    #print(G.edges[(0,0),(1,0)])
    cval = torch.tensor(inp, requires_grad = True, dtype=torch.float32)
    vals = [cval]

    relu = torch.nn.ReLU()

    #print('Cval: ', cval)
    #print('Cval req grad: ', cval.requires_grad)

    # Evaluate inner layers
    # for w, b in zip(net.weights[:-1], net.biases[:-1]):
    #     cval = relu(cval @ torch.from_numpy(w) + torch.from_numpy(b))
    #     vals.append(cval)

    weights = []
    biases  = []

    #print(len(layers)-1)
    for i in range(len(layers)-1):
        weight_matrix_lyr = []
        bias_matrix_lyr = []
        for y in range(len(layers[i])):
            weight_neuron = []
            # bias_neuron = []
            node = layers[i][y]  
            adj = G.adj[node]
            #print(node)
            for x in adj:            
                w = G.edges[ node, x]['weight']
                weight_neuron.append(w)            
            if i == 0:
                bias_matrix_lyr.append(0)             
            else:
                bias_matrix_lyr.append(G.nodes[node]['bias']) 
            weight_matrix_lyr.append(weight_neuron)
        weights.append(weight_matrix_lyr)
        biases.append(bias_matrix_lyr)
    
    bias_lst_lyr = []
    
    for y in range(len(layers[len(layers)-1])):
        node = layers[len(layers)-1][y] 
        bias_lst_lyr.append(G.nodes[node]['bias'])
    biases.append(bias_lst_lyr)

    for w, b in zip(weights[:-1], biases[1:-1]):
        cval = relu(cval @ torch.from_numpy(np.array(w, dtype=np.float32)) + torch.from_numpy(np.array(b, dtype=np.float32)))
        vals.append(cval)
                   
    # Evaluate last layer
    cval = cval @ torch.from_numpy(np.array(weights[-1], dtype=np.float32)) + torch.from_numpy(np.array(biases[-1], dtype=np.float32))
    if end_relu:
        cval = relu(cval)
    vals.append(cval)

    vals[-1][0].backward(inputs = vals)

    grads = [ v.grad.numpy() for v in vals ]

    imp_neu = torch.mul(torch.tensor(inp, dtype=torch.float32), torch.from_numpy(grads[0]))

    
    important_neurons = []
    for i,val in enumerate(imp_neu):
        tuple_neu = ((0,i),val.item())
        important_neurons.append(tuple_neu)

    important_neurons = sorted(important_neurons, key=lambda x: x[1])


    #print('grads: ', (important_neurons))

    return important_neurons


if __name__=="__main__":
    G = reader.network_data("../networks/path_to_save_model.onnx")
    G1 = copy.deepcopy(G)
    f = open("../images/5img36453", "r")
    image = ast.literal_eval(f.read())
    class_of_op = utils.findClassForImage(G,image)
    print("class_of_op", class_of_op)
    conj_lin_equations = [
        [-1,0,0,0,0,1,0,0,0,0,0],
        [0,-1,0,0,0,1,0,0,0,0,0],
        [0,0,-1,0,0,1,0,0,0,0,0],
        [0,0,0,-1,0,1,0,0,0,0,0],
        [0,0,0,0,-1,1,0,0,0,0,0],
        [0,0,0,0,0,1,-1,0,0,0,0],
        [0,0,0,0,0,1,0,-1,0,0,0],
        [0,0,0,0,0,1,0,0,-1,0,0],
        [0,0,0,0,0,1,0,0,0,-1,0]
    ]
    property_encode.encoding_property(G,False,conj_lin_equations)
    #print(G.nodes())
    imp_neus = compute_gradient(G,image,False)
    img_dict = {}
    img_dict.update({(0, i): val for i, val in enumerate(image)})


    important_neurons = []
    [important_neurons.append((i[0],img_dict[i[0]])) for i in imp_neus]

    #print(important_neurons)

    inp_lb = [0]*784
    inp_ub = [1]*784
    
    #inputs = [node for node in G.nodes() if node[0] == 0]
    # sorted(inputs)
    # inp_features = [(node,image[i]) for i,node in enumerate(inputs)]
    # inp_features = sorted(inp_features, key=lambda x: x[1])
    # print("inp_features", inp_features)
    explaination= contrastive.find_explanation(important_neurons,G,inp_lb,inp_ub)
    image = Image.new("L", (28, 28), color=255)
    draw = ImageDraw.Draw(image)
    for neuron in explaination:
    	nid = neuron[0]
    	lyr = nid[0]
    	neu = nid[1]
    	row_ind = neu//28
    	col_ind = neu - (neu//28)*28
    	draw.point((row_ind, col_ind), fill=100)
    # 	#img[row_ind][col_ind] = neuron[1]   
    	

    print("explaination",explaination)	
    image.save("output.png")
    # #image.show()





    # print("length", len(explaination))
    # #print("ub_by_lb",ub_by_lb)

