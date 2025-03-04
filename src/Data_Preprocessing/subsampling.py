from networkx.readwrite import graphml
from preprocess import pre_process_aml_world,pre_process_rabobank,pre_process_saml_d, pre_process_elliptic
from utils import *

ego_node_dimension = 2

'''
G_aml_world = pre_process_aml_world()
G_aml_world_undirected = G_aml_world.to_undirected()
#check if selecting random nodes works
nodes = select_nodes(G_aml_world, 75, True)
all_nodes = list(G_aml_world.nodes())

for node in G_aml_world.nodes:
    call_sampling_probability(G_aml_world_undirected, node, ego_node_dimension)
'''

G_test_aml_world = pre_process_aml_world()
G_test_aml_rabobank = pre_process_rabobank()
G_test_saml_D = pre_process_saml_d()
G_test_elliptic = pre_process_elliptic()
print('done')
print('hi')
print('done')

