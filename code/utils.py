import networkx as nx
def getLayers(G):
    """
    Given a graph G, collect nodes in each layer into a list.

    NOTE: If some layers are empty, they will be skipped in the returned list.
    So, if the nodes are (0,0), (0,1), (2,0), then the returned list will be
    [[(0,0), (0,1)], [(2,0)]]

    Returns a list of list of nodes, one such list for each layer. Each list for
    each layer is sorted.
    """
    layer_dict = {}     # arrange nodes in G according to their layer
    for node in G.nodes:

        if node[0] not in layer_dict:
            layer_dict[node[0]] = []
        layer_dict[node[0]].append(node)

    layers = []
    for i in sorted( layer_dict.keys() ):
        layers.append( layer_dict[i] )
    for l in layers: l.sort()

    return layers

def getLayerSize(layers):
    """
    Take the mapping of which nodes are present in which layers and return the number of neurons in each layers
    """
    G_layer_sizes = []
    for i in layers:
        G_layer_sizes.append(len(i))
    return G_layer_sizes

def findClassForImage(G,val):
    simulationDict = {}
    layers = getLayers(G)
    for i,j in enumerate(val):
        simulationDict[(0,i)] = j

    H = nx.reverse_view(G)

    for i in range(len(layers)-1):
        for y in range(len(layers[i+1])):
            node = layers[i+1][y]
            #print(node)
            value_node = 0   
            adj = H.adj[node]
            for x in adj:            
                w = G.edges[x, node]['weight']

                value_node += simulationDict[x]*w

            value_node += G.nodes[node]['bias'] 

            if((i+1)!=(len(layers)-1)):  
                value_node = max( value_node, 0 )  
                   
            simulationDict[node] = value_node 

    last_layer = layers[-1]
    last_layer_simul = {node :simulationDict[node] for node in last_layer}
    max_key = max(last_layer_simul, key=lambda k: int(last_layer_simul[k]))

    #print("last_layer_simul", last_layer_simul)

    return max_key

def computeValForNetwork(G,val):
    simulationDict = {}
    layers = getLayers(G)
    for i,j in enumerate(val):
        simulationDict[(0,i)] = j

    H = nx.reverse_view(G)

    for i in range(len(layers)-1):
        for y in range(len(layers[i+1])):
            node = layers[i+1][y]
            value_node = 0   
            adj = H.adj[node]
            for x in adj:            
                w = G.edges[x, node]['weight']

                value_node += simulationDict[x]*w

            value_node += G.nodes[node]['bias'] 

            if((i+1)!=(len(layers)-1)):  
                value_node = max( value_node, 0 )  
                   
            simulationDict[node] = value_node 


    return simulationDict
