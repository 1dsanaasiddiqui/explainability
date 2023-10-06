import utils
import networkx as nx
import sys
import property_encode
import itertools
import concurrent.futures
sys.path.append( "../Marabou/" )
from maraboupy import Marabou, MarabouCore


LB = 0
UB = 784
free = set()
singleton = set()
pairs = set()

def find_singleton(ip_features,G,inp_low,inp_upp):
    """
    Inputs:
    ip_features : The inputs with their original values in the original image
    G : the Network with the original property encoded
    inp_low : The lower bounds on inputs
    inp_upp : The upper bounds on inputs
    Returns : postions where single pixel can modify the  label and postion where single pixel modification cannot
    lead to an attack
    """
    global singleton
    fixed_features = set(ip_features)
    for ip_f in ip_features:
            fixed_features.remove(ip_f)
            tuple_cex =  verify(fixed_features,[ip_f],G,inp_low,inp_upp)
            if tuple_cex[0] == 'SAT':
                    print("Singleton , ip",tuple_cex[1],ip_f[0])
                    singleton.add(ip_f)
                    LB = LB+1

            else:
            	print("Not Singleton , ip",tuple_cex[1],ip_f[0])
            fixed_features.add(ip_f)
            


def verify(fixed_features,free_features,G,inp_low,inp_upp):
    """
    Encode the merged graph into a Marabou query and attempt to verify it using
    Marabou.

    Arguments:
    fixed_features - The features whose values cannot be changed
    free_features  - The who can take any value in their domain
    G           -   The graph for the network
    inp_lb      -   The lower bound on the inputs
    inp_ub      -   The upper bound on the inputs
    Reuturns:
    A counterexample and Sat is there is an assignment, None otherwise
    """
    #print("FREE", free_features)
    # Get the input nodes and output node
    layers = utils.getLayers( G )
    inp_nodes = layers[0]
    assert len( layers[-1] ) == 1
    out_node = layers[-1][0]
    # Create variables for forward and backward
    n2v_post = { n : i for i, n in enumerate( G.nodes() ) }
    n2v_pre = {
        n : i + len(n2v_post)
        for i, n in enumerate( itertools.chain( *layers[1:] ))
    }

    # Reverse view
    rev=nx.reverse_view(G)

    # Set up solver
    solver = MarabouCore.InputQuery()
    solver.setNumberOfVariables( len(n2v_post) + len(n2v_pre) )
    # Encode the network

    for node in G.nodes():
        eq = MarabouCore.Equation()
        flag = False
        for pred in rev.neighbors(node):
            flag = True
            a = G.edges[(pred,node)]['weight']
            eq.addAddend(a, n2v_post[pred])
        if flag:  #and G.neighbors(node)!=[]:
            eq.addAddend(-1, n2v_pre[node])
            eq.setScalar(-1*G.nodes[node]['bias'])
            solver.addEquation(eq)
            if(node!=out_node):
                MarabouCore.addReluConstraint(solver,
                        n2v_pre[node], n2v_post[node])
            else:
                eq1 = MarabouCore.Equation()
                eq1.addAddend(1,n2v_pre[out_node])
                eq1.addAddend(-1,n2v_post[out_node])
                eq1.setScalar(0)
                solver.addEquation(eq1)

    # Encode precondition
    for feature in fixed_features:
        node,val = feature[0],feature[1]
        solver.setLowerBound(n2v_post[node],val)
        solver.setUpperBound(n2v_post[node],val)

    for feature in free_features:
    	
        node = feature[0]
        #print("Node", feature[1])
        solver.setLowerBound(n2v_post[node],inp_low[node[1]])
        solver.setUpperBound(n2v_post[node],inp_upp[node[1]])

        
    # Encode postcondition
 
    solver.setUpperBound( n2v_post[ out_node ], 0)

    options=Marabou.createOptions(
        verbosity = 0 )
    ifsat, var, stats = MarabouCore.solve(solver,options,'')
    if(len(var)>0):
        ycex=var[n2v_post[out_node]]
        cex=[]
        for node in inp_nodes:
            cex.append((node, var[n2v_post[node]]))

        #if config.DEBUG:
        #    utils.log("Out (4,0): ", var[ n2v_post[ (4,0) ]])

        return ('SAT',cex)
    else:
        return  ('UNSAT',None)
    
def upper_bounding_thread(ip_features,G,ip_low,ip_upp):
    #global singleton
    global free
    global pairs
    global UB
    Explanation = set(ip_features) 
    #print("HERE")


    #remaining_features = list(set(ip_features) - singleton)
    remaining_features = ip_features
    remaining_features1 = ip_features
    L = 0
    R = len(remaining_features)-1
    while L <= len(remaining_features)-1:
        while L <= R:
            mid = (L+R)//2
            Explanation = set(remaining_features)-free
            to_remove = set(remaining_features[L:mid+1])
            print("Checking for ", to_remove)
            t = list(free)
            # print(free)
            t.extend(to_remove)
            #print(t)
            tuple_cex = verify(Explanation-to_remove,t ,G,ip_low,ip_upp)
            if tuple_cex[0] == 'UNSAT':
                free = free.union(to_remove) 
                print("free",to_remove)
                UB = UB- len(to_remove)
                L = mid+1
            else:
                print("Not free",to_remove)
                # cex = tuple_cex[1]
                # local_singletons = []
                # for i, f in enumerate(remaining_features[mid+1:]):
                #     x = (f[0],f[1])
                #     z = list()
                #     z.append(x)
                #     print("Local Singleton Search", z)
                #     tuple_cex = verify(set(cex)-set(f) ,z ,G,ip_low,ip_upp)
                #     if tuple_cex[0] == 'SAT':
                #         print("Not free", z)
                #         remaining_features.pop(i)             
                R= mid-1

        L  = L + 1
        R  = len(remaining_features)-1

    return Explanation


def lower_bounding_thread(ip_features,G,inp_low,inp_upp):
	
	constrastive_pairs(ip_features,G,inp_low,inp_upp)
	

	
	
def find_explanation(inp_features,G,inp_lb,inp_ub):
	#global singleton
	#find_singleton(inp_features,G,inp_lb,inp_ub)
	#print("Singleton", singleton)
	#with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
	upper_bounding_thread(inp_features,G,inp_lb,inp_ub)
	    
	    #future2 = executor.submit(lower_bounding_thread(inp_features,G,inp_lb,inp_ub))
	    
	# Return the required values
	result_F = set(inp_features) - free
	result_UB = UB
	result_LB = LB
	return result_F


     
def constrastive_pairs(ip_features,G,inp_low,inp_upp):
    """
    Inputs:
    ip_features : The inputs with their original values in the original image
    G : the Network with the original property encoded
    inp_low : The lower bounds on inputs
    inp_upp : The upper bounds on inputs
    Returns : postions where two pixels can modified to change the  label and postions where 2 pixel modification cannot lead to an attack
    """
    global pairs
    ip_features = set(ip_features) - singleton
    ip_pairs = list(itertools.combinations(ip_features, 2))
    fixed_features = set(ip_features)
    for pair in ip_pairs:
            #print("Feature considered",pair)
            feature_1 =  pair[0]
            feature_2 = pair[1]
            if feature_1 in fixed_features:
            	fixed_features.remove(feature_1)
            if feature_2 in fixed_features:
            	fixed_features.remove(feature_2)
            tuple_cex =  verify(fixed_features,[feature_1,feature_2],G,inp_low,inp_upp)
            if tuple_cex[0] == 'SAT':
                    print("Pairs",pair)
                    paris.add(pair)
            else:
            	print("Not Pairs",pair)
            	
            
            fixed_features.add(feature_1)
            fixed_features.add(feature_2)
    



if __name__ == "__main__":


    G=nx.DiGraph()
    G.add_nodes_from([ (0,0),(0,1),(0,2) ])
    G.add_nodes_from([ ((1,0),{'bias':-2.}),((1,1),{'bias':-3.}),((1,2),{'bias':0.}) ])
    G.add_nodes_from([ ((2,0),{'bias':0.}),((2,1),{'bias':0.})])
    G.add_edges_from([ ( (0,0), (1,0), {'weight': 2.} ), ( (0,0), (1,1), {'weight': 2.} ) , ( (0,0), (1,2), {'weight': 2.} ),])
    G.add_edges_from([ ( (0,1), (1,0), {'weight': 3.} ), ( (0,1), (1,1), {'weight': 3.} ) , ( (0,1), (1,2), {'weight': 3.} ),])
    G.add_edges_from([ ( (0,2), (1,0), {'weight': 5.} ), ( (0,2), (1,1), {'weight': 7.} ) , ( (0,2), (1,2), {'weight': 6.} ),])
    G.add_edges_from([ ( (1,0), (2,0), {'weight': 1.} ), ( (1,0), (2,1), {'weight':-1.})])
    G.add_edges_from([ ( (1,1), (2,0), {'weight': 1.} ), ( (1,1), (2,1), {'weight':-1.})])
    G.add_edges_from([ ( (1,2), (2,0), {'weight': 0.} ), ( (1,2), (2,1), {'weight':1.})])
    conj_lin_equations = [ [1,-1,0]
    ]
    property_encode.encoding_property(G,False,conj_lin_equations)
    inp_features = [ ((0,0),0),((0,1),1),((0,2),1) ]
    inp_lb = [0,0,0]
    inp_ub = [1,1,1]
    Explanation = find_explanation(inp_features,G,inp_lb,inp_ub)
    print("Explanation",Explanation)
 
