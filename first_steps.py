
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.spatial import Delaunay

# extra required packages to implement own mutation algorithm in DEAP
from itertools import repeat
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

#%%
# Read .xlsx file with pandas
def read_preprocess_data(fname):
    """
    Read .xlsx file of trees and create the network of trees from it.

    Parameters
    ----------
    fname : str
        path to .xlsx file

    Returns
    -------
    data : pandas dataframe
        Xontent of .xlsx file
    graph : NetworkX graph
        Undirected graph with node attributes (positio and tree species)

    """
    data = pd.read_excel(fname)
    # Convert pandas DataFrame to a NetworkX graph variable.
    #   We need to specify the name of the comlumns that will be used as nodes.
    #   NetworkX will create a graph with source nodes and target nodes, and
    #   by selecting those to be the same column from the DataFrame, we create 
    #   a graph of only self-loops. 
    data['OBJECTID'] = data['OBJECTID']-1
    graph = nx.from_pandas_edgelist(data, source='OBJECTID', target='OBJECTID')
    graph.remove_edges_from(nx.selfloop_edges(graph)) # remove selfloops
    
    # Give extra attributes to nodes.
    nx.set_node_attributes(G=graph, values=data['x'].to_dict(), name='x')
    nx.set_node_attributes(G=graph, values=data['y'].to_dict(), name='y')
    nx.set_node_attributes(G=graph, values=data['maintree_hmax'].to_dict(), name='species')
    
    return data, graph

def draw_forest(G):
    dict_of_positions = {}
    for item in nx.get_node_attributes(G, 'x').items():
        dict_of_positions[item[0]] = (item[1], nx.get_node_attributes(G, 'y')[item[0]])
    plt.figure()
    nx.draw(G, pos=dict_of_positions, node_size=5, node_color='green', edgelist=graph.edges)
    return 0

def delaunay_triangulation(graph, removed_nodes=[]):
    """
    Generates Delaunay triangulation and returns  the (unweighted) edge list

    Parameters
    ----------
    graph : networkx graph
    
    removed_nodes : list (optional, default: [])
        List of nodes that are not part of the graph.
        This list is necessary to manually make the Delaunay triangulation algorithm 
        keep track of node names.

    Returns
    -------
    edge_list : NumPy array
        Unweighted list of edges

    """
    pos_xy = [(node[1]['x'], node[1]['y']) for node in np.array(graph.nodes(data=True))]
    tri = Delaunay(np.array(pos_xy)) 
    # tri.simplices gives (node1, node2, node3) triangles. have to convert that into edges (node1, node2)
    triangles = tri.simplices 
    edges1 = triangles[:,0:2]
    edges2 = triangles[:,1:3]
    edges3 = np.delete(triangles, 1, axis=1)
    
    # TODO: CHECK FOR DUPLICATES! (A,B) , (B,A)
    edge_list = np.vstack((edges1, edges2, edges3))
    
    # Move name of node forward by 1 every removed node
    for rn in sorted(removed_nodes):
        edge_list = edge_list + 1 * (edge_list >= rn)
    
    return edge_list

def generate_edge_weights(G, edge_list):
    """
    Compute edge weights from a list of edges

    Parameters
    ----------
    G : nx graph
    
    edge_list : NumPy array
        List of edges (pairs of nodes)

    Returns
    -------
    edge_list_weight : list
        Weighted list of edges

    """
    edge_list_weight = []
    for node_pair in edge_list:
        n1 = G.nodes[node_pair[0]]
        n2 = G.nodes[node_pair[1]]
        dist = (n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2
        weight = 100 * 1/dist
        edge_list_weight.append([node_pair[0], node_pair[1], weight])
    
    return edge_list_weight

def walk_eigen(graph):
    """
    Ordered eigenvalues and eigenvectors of the walk matrix

    Parameters
    ----------
    graph : networkx graph
    Raises
    ------
    ValueError
        If mismatch between 1st eigenvalue and direct measurement of graph connectedness.

    Returns
    -------
    eigvals : np.array
        Ordered eignevalues of the walk matrix, from highest to lowest
    eigvects : np.array
        Eigenvectors corresponding to the eigenvalues.

    """
    n_nodes = len(graph.nodes)
    M = nx.to_numpy_array(graph)
    d = np.sum(M,axis=0)
    Dinv = 1/d * np.identity(n_nodes)
    
    W = np.matmul(M, Dinv) # non-lazy RW because sum(weights) is not a normalized probability
    
    eigvals, eigvects = np.linalg.eig(W)
    # sort eigenvalues and eigenvectors
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvects = eigvects[:,idx]
    
    # quick check of conectivity
    if nx.is_connected(graph) != (abs(1.0 - eigvals[0]) < 1e-5):
        raise ValueError('graph connectivity is not reflected in first eigenvalue')
    
    return eigvals, eigvects


#%%    
data, graph_original = read_preprocess_data(fname = 'potentialTreeList_v1.xlsx')

n_nodes = len(data)

# Every graph below should be a deepcopy of the original.
graph = copy.deepcopy(graph_original)

edge_list = delaunay_triangulation(graph, removed_nodes=[]) # triangulate
# add weighted edges      
graph.add_weighted_edges_from(generate_edge_weights(graph, edge_list), weight='weight')

# Draw t!
draw_forest(graph)

eigvals, eigvects = walk_eigen(graph) # Walk matrix eigval and eigvect are computed like this


#%%
# Remove nodes one by one and check change in parameters
NODE_REMOVALS = 3

# 0th iteration
edge_list = delaunay_triangulation(graph, [])        
graph.add_weighted_edges_from(generate_edge_weights(graph, edge_list), weight='weight')
eigvals, eigvects = walk_eigen(graph)

nodes_removed = []
eigv1 = []
eigv2 = []

for i in range(0, NODE_REMOVALS):
    eigv1.append(eigvals[0])
    eigv2.append(eigvals[1])
    
    node_to_remove = np.random.choice(graph.nodes)
   
    print(f'>>>>>>>> NODE TO REMOVE = {node_to_remove}')
    graph.remove_node(node_to_remove) # remove node
    
    nodes_removed.append(node_to_remove)
    
    # TODO: Possible incremental triangulation? -> Only check triangles that were not there before -> more efficient
    edge_list = delaunay_triangulation(graph, nodes_removed) # triangulate       
    graph.add_weighted_edges_from(generate_edge_weights(graph, edge_list), weight='weight') # add weights
    
    eigvals, eigvects = walk_eigen(graph) # compute eigval

    print(i)
    
plt.figure()
plt.plot([i for i in range(0, NODE_REMOVALS)], eigv2, 'o-', label='$\lambda_2$')

#%%
# Genetic Algorithm stuff
import random
from deap import creator, base, tools, algorithms
import multiprocessing


N_CPUS = 4
N_TREES_TO_CUT = 10
n_nodes = len(data)

creator.create("FitnessMin", base.Fitness, weights=(+1.0,)) # if weights are positive, we have maximization. They must be a tuple. In DEAP, single objective minimization is a pecial case of multiobjective
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, n_nodes-1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, n=N_TREES_TO_CUT)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



def _mutation_check_duplicates(individual, low, up, indpb):
    """Fork of deap.tools.mutUniformInt, but not allowing duplicated ints in the 
    same individual
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            int_to_add = random.randint(xl, xu)
            if int_to_add not in individual:
                individual[i] = int_to_add
    
    if len(set(individual)) != len(individual):
        print("duplication exists!")

    return individual,

def individual_points_per_species(G, points_per_species):
    pointlist = [points_per_species[tree['species']] for i, tree in G.nodes(data=True)]
    return np.sum(pointlist)

def biodiversity(individual): # this should be returning the biodiversity value in a tuple
    
    graph = copy.deepcopy(graph_original) 
    
    graph.remove_nodes_from(individual) # remove nodes

    edge_list = delaunay_triangulation(graph, removed_nodes=individual) # triangulate
    edge_weights = generate_edge_weights(graph, edge_list)       
    graph.add_weighted_edges_from(edge_weights, weight='weight') # add weights
    eigvals, eigvects = walk_eigen(graph) # compute eigval of walk matrix
    
    l2 = eigvals[1] # the smaller the 2nd eigenvalue, the faster a random walker visits all nodes
    sum_of_weights = graph.size(weight='weight')
    
    individual_sum = individual_points_per_species(graph,
                                                   points_per_species={'Pine': 1, 'Spruce': 2, 'Deciduos': -0.1})
    
    # TODO: add node attributes of tree species and heights, then biodiversity depends on those as well
    biodiv = 0*sum_of_weights + 10000 * 1/l2 + 0.1*individual_sum # silly biodiversity function

    return biodiv,

toolbox.register("evaluate", biodiversity)
toolbox.register("mate", tools.cxOnePoint) # single point crossover
toolbox.register("mutate", _mutation_check_duplicates, low=0, up=n_nodes-1, indpb=0.1) # replaces individual's attribute with random int
toolbox.register("select", tools.selTournament, tournsize=int(N_TREES_TO_CUT/3)+1)



def GA(N_GENERATIONS, N_CPUS):
#    random.seed(64)
    
    if N_CPUS >1:
        pool = multiprocessing.Pool(processes=N_CPUS)
        n_population = N_CPUS
        toolbox.register("map", pool.map)
    else:
        n_population=5
    
    pop = toolbox.population(n=n_population)
    
    def _check_duplicates_in_population(pop):
        dupl = False
        for n in range(n_population):
            if len(pop[n]) != len(set(pop[n])):
                dupl = True
                pop[n] = [random.randint(0, n_nodes-1) for i in range(N_TREES_TO_CUT)]
        return dupl, pop
    
    # Identify possible duplicates and generate new starting population if duplicated
    habemus_duplicate = True
    while habemus_duplicate:
        habemus_duplicate, pop = _check_duplicates_in_population(pop)        
           
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.1, ngen=N_GENERATIONS, 
                        stats=stats, halloffame=hof, verbose=True)
    
    if N_CPUS > 1:
        pool.close()

    best_ind = tools.selBest(pop, 1)[0]
    
    return best_ind, pop, log
    

best_individual, pop, log = GA(100,1)    
