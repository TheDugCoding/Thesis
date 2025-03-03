from networkx.readwrite import graphml
from preprocess import pre_process_aml_world
from utils import *

ego_node_dimension = 2

G_aml_world = pre_process_aml_world()
G_aml_world_undirected = G_aml_world.to_undirected()
#check if selecting random nodes works
nodes = select_nodes(G_aml_world, 75, True)
all_nodes = list(G_aml_world.nodes())

for node in G_aml_world.nodes:
    call_sampling_probability(G_aml_world_undirected, node, ego_node_dimension)




