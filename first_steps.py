
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read .xlsx file with pandas

# If the data is in the same folder as the python script, use this:
fname = 'potentialTreeList_v1.xlsx'
# If the data is somewhere else, specify the whole path:
# fname = r'C:\Users\03125327\Desktop\potentialTreeList_v1.xlsx' # absolute position of the .xlsx file.
data = pd.read_excel(fname)

# Convert pandas DataFrame to a NetworkX graph variable.
#   We need to specify the name of the comlumns that will be used as nodes.
#   NetworkX will create a graph with source nodes and target nodes, and
#   by selecting those to be the same column from the DataFrame, we create 
#   a graph of only self-loops. 
graph = nx.from_pandas_edgelist(data, source='OBJECTID', target='OBJECTID')

# Give extra attributes to nodes. Optional, not used later
nx.set_node_attributes(G=graph, values=data['x'].to_dict(), name='x')
nx.set_node_attributes(G=graph, values=data['y'].to_dict(), name='y')

pos_x_y = list(zip(data['x'].values, data['y'].values)) #list of (x,y) coordinate tuples.

# Create a dictionary that has the form {node: (coordx, coordy)}. This is required to draw the graph.
dict_of_positions = {}
for i in range(1, 1061+1):
    dict_of_positions[i] = pos_x_y[i-1]

# Draw it!
nx.draw(graph, pos=dict_of_positions)