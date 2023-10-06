import utils
import networkx as nx
import numpy as np

def add_nodes_in_second_last_layer(G,last_layer,conj_lin_equations):
    """
    Adds nodes in the second last layer to convert it into a boolean property
    """
    nodes_in_second_last_layer = []
    for i in range(len(conj_lin_equations)):
        #print((last_layer+1,i))
        nodes_in_second_last_layer.append((last_layer+1,i))
        G.add_node((last_layer+1,i),bias = -1*conj_lin_equations[i][-1])
    return nodes_in_second_last_layer

def add_nodes_in_last_layer(G,last_layer,count):
    """
    Add node in the  last layer to convert it into a boolean property
    """
    G.add_node((last_layer+2,0),bias=0)
    node = (last_layer+2,0)
    for i in range(count):
        G.add_edge((last_layer+1,i),node,weight=1)

def encoding_property(G,flag_last_reLU,conj_lin_equations):

    """
    Encodes given conjunction of linear equations into the network. Note that
    the property is encoded as if it were a SAT, that is, if the original
    network has an input that satisfies conj_lin_equations, then the encoded
    networks output will be <= 0 for that point. On the other hand, if 0 is a
    strict upper bound for encoded network, we know that there is no input of
    the network that satisfies conj_lin_equations.

    G                   : graph to be encoded
    conj_lin equations  : it is a list of list which specify the constraints in
                          form of inequalities given by the terms up to the last
                          term, and the upperbound term is the last term
    """

    # Set up helpers
    layers = utils.getLayers(G)
    last_layer_node = layers[-1]
    nodes_in_n_1_layer = layers[len(layers)-2]
    #print(nodes_in_n_1_layer)
    last_layer = last_layer_node[0][0]

    # Encode as if last layer has relu
    nodes_in_second_last_layer=add_nodes_in_second_last_layer(G,last_layer,conj_lin_equations)
    count = 0

    for i in conj_lin_equations:
        node = (last_layer+1,count)

        for j in range(len(i)-1):
            if(i[j]!=0):

                G.add_edge(last_layer_node[j],nodes_in_second_last_layer[count],weight=i[j])

        count = count+1

    if(count>1):
        add_nodes_in_last_layer(G,last_layer,count)

    layers = utils.getLayers(G)
    layer_sizes = utils.getLayerSize(layers)

    if not flag_last_reLU:

        # Set up empty weights
        weights_n_1_last = np.zeros(
            ( len(nodes_in_n_1_layer), len(last_layer_node) ),
            dtype = np.float64,
        )
        weights_last_second_last = np.zeros(
            ( len(last_layer_node), len(nodes_in_second_last_layer) ),
            dtype = np.float64,
        )

        # Fill weight matrices
        for i, ni in enumerate(nodes_in_n_1_layer):
            for j, nj in enumerate(last_layer_node):
                if G.has_edge(ni,nj):
                    weights_n_1_last[i][j] = G.edges[ ni, nj ][ 'weight' ]
        for i, ni in enumerate(last_layer_node):
            for j, nj in enumerate(nodes_in_second_last_layer):
                if G.has_edge(ni,nj):
                    weights_last_second_last[i][j] = G.edges[ ni, nj ][ 'weight' ]

        # Set up biases
        bias_last = np.zeros( ( len(last_layer_node), ), dtype = np.float64 )
        bias_second_last = np.zeros(
            ( len(nodes_in_second_last_layer), ), dtype = np.float64
        )

        # Fill biases
        for i, n in enumerate( last_layer_node ):
            bias_last[i] = G.nodes[ n ][ 'bias' ]
        for i, n in enumerate( nodes_in_second_last_layer ):
            bias_second_last[i] = G.nodes[ n ][ 'bias' ]

        # New weights and biases
        weights_n_1_second_last = weights_n_1_last @ weights_last_second_last
        biases_new = bias_second_last + bias_last @ weights_last_second_last

        # Creates new edges with correct weight
        for i, ni in enumerate(nodes_in_n_1_layer):
            for j, nj in enumerate(nodes_in_second_last_layer):
                if(weights_n_1_second_last[i][j]!=0):
                    G.add_edge( ni, nj, weight=weights_n_1_second_last[i][j] )

        # Set up biases
        for i, n in enumerate( nodes_in_second_last_layer ):
            G.nodes[ n ][ 'bias' ] = biases_new[i]

        # Remove equation layer
        for i in last_layer_node:
            G.remove_node(i)

if __name__ == "__main__":
    G1 = nx.DiGraph()
    G1.add_nodes_from([ (0, 0), (0, 1)])
    G1.add_nodes_from([
        ( (1, 0), {'bias': 0.} ),
        ( (1, 1), {'bias': 0.} ),
        ( (1, 2), {'bias': -1.} ),
        ( (1, 3), {'bias': -1.} ),
        ( (1, 4), {'bias': 0.} ),
        ( (1, 5), {'bias': 0.} ),
        ( (1, 6), {'bias': -1.} ),
        ( (1, 7), {'bias': -1.} )])
    G1.add_nodes_from([ ( (2, 0), {'bias': 1.} ), ( (2, 1), {'bias': 2.} ),
                        ( (2, 2), {'bias': 3.} ), ( (2, 3), {'bias': 4.} )])

    G1.add_edges_from([
        ( (0, 0), (1, 0), {'weight': 1000.} ),
        ( (0, 0), (1, 1), {'weight': -1000.} ),
        ( (0, 0), (1, 2), {'weight': 1000.} ),
        ( (0, 0), (1, 3), {'weight': -1000.} )  ])
    G1.add_edges_from([
        ( (0, 1), (1, 0), {'weight': -1000.} ),
        ( (0, 1), (1, 1), {'weight': 1000.} ),
        ( (0, 1), (1, 2), {'weight': -1000.} ),
        ( (0, 1), (1, 3), {'weight': 1000.} )  ])
    G1.add_edges_from([
        ( (0, 1), (1, 4), {'weight': 1000.} ),
        ( (0, 1), (1, 5), {'weight': -1000.} ),
        ( (0, 1), (1, 6), {'weight': 1000.} ),
        ( (0, 1), (1, 7), {'weight': -1000.} )  ])
    G1.add_edges_from([
        ( (0, 0), (1, 4), {'weight': -1000.} ),
        ( (0, 0), (1, 5), {'weight': 1000.} ),
        ( (0, 0), (1, 6), {'weight': -1000.} ),
        ( (0, 0), (1, 7), {'weight': 1000.} )  ])

    G1.add_edges_from([ ( (1, 0), (2, 0), {'weight': 1.} ), ( (1, 2), (2, 0), {'weight': -1.} ) ])
    G1.add_edges_from([ ( (1, 1), (2, 1), {'weight': 1.} ), ( (1, 3), (2, 1), {'weight': -1.} ) ])
    G1.add_edges_from([ ( (1, 4), (2, 2), {'weight': 1.} ), ( (1, 6), (2, 2), {'weight': -1.} ) ])
    G1.add_edges_from([ ( (1, 5), (2, 3), {'weight': 1.} ), ( (1, 7), (2, 3), {'weight': -1.} ) ])


    G1_layer_sizes = [ 2, 8, 4 ]
    length  = len(G1_layer_sizes)
    # utils.print_G(G1,G1_layer_sizes)
    # y(2,3)-y(2,0) <= 0
    # y(2,2)-y(2,1) <= 0


    equations = [
        [1,-1,0,0,1],
        [1,0,-1,0,2],
        [1,0,0,-1,3]
    ]
    encoding_property(G1,False,equations)

    print(G1.nodes(data= True))
    print(G1.edges(data=True))

