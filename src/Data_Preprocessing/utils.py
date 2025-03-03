import networkx as nx
from networkx.readwrite import graphml
import numpy as np



def get_structural_info(G):
    '''
    This function compute several structural node measurements, if a ne
    :param G: A Graph in networkX
    :return: A graph G, containing additional structural information
    '''
    nx.set_node_attributes(G, dict(nx.degree(G)), 'degree')
    nx.set_node_attributes(G, nx.degree_centrality(G), 'degree_centrality')
    nx.set_node_attributes(G, nx.pagerank(G, alpha=0.85), 'pagerank')

    return G

def select_nodes(preprocessed_dataset: nx.DiGraph, percentage_of_node_to_sample_per_dataset = 100, select_random_nodes = True):
    '''
    :param preprocessed_dataset: dataset containing preprocessed data
    :param percentage_of_node_to_sample_per_dataset: percentage of nodes to sample per dataset
    :param select_random_nodes: the nodes of the dataset will be selected randomly
    :return:
    '''

    nodes = list(preprocessed_dataset.nodes())
    total_nodes = len(nodes)
    num_nodes_to_sample = int((percentage_of_node_to_sample_per_dataset / 100) * total_nodes)
    return np.random.choice(nodes, num_nodes_to_sample, replace=False)


def extract_features(G, node):
    """
    Extracts features (pagerank, degree_centrality, degree) for a given node.

    :param G: NetworkX DiGraph
    :param node: Node ID
    :return: NumPy array of node features
    """
    if node not in G.nodes:
        raise ValueError(f"Node {node} not found in the graph.")

    return np.array([
        G.nodes[node].get("pagerank", 0),          # Get pagerank, default 0 if missing
        G.nodes[node].get("degree_centrality", 0), # Get degree centrality
        G.nodes[node].get("degree", 0)             # Get degree
    ])

def call_sampling_probability(G, center_node, M):
    """
    Calls sampling_probability for all nodes in the M-hop neighborhood of a given center node.

    :param G: NetworkX DiGraph
    :param center_node: The central node for the M-hop neighborhood
    :param M: The hop distance
    :return: Dictionary of nodes with their sampling probability
    """
    if center_node not in G.nodes:
        raise ValueError(f"Center node {center_node} not found in the graph.")

    # Get M-hop neighborhood (including center_node)
    ego_nodes = list(nx.ego_graph(G, center_node, radius=M, undirected=True).nodes)

    # Extract feature vector of center node
    center_features = extract_features(G, center_node)

    # Extract feature matrix of all neighborhood nodes (excluding center)
    neighborhood_features = np.array([
        extract_features(G, n) for n in ego_nodes if n != center_node
    ])

    # Compute sampling probabilities
    sampling_probs = {}
    for vi in ego_nodes:
        if vi == center_node:
            continue  # Skip center node itself

        vi_features = extract_features(G, vi)  # Get feature vector for node vi
        prob = sampling_probability(vi_features, center_features, M, neighborhood_features)
        sampling_probs[vi] = prob

    return sampling_probs


def sampling_probability(vi_features: np.ndarray, center_features: np.ndarray, m: int,
                         neighborhood_features: np.ndarray) -> float:
    """
    Computes the probability PR(vi, c) for a node vi being sampled from the M-hop neighborhood.

    Parameters:
    - vi_features (np.ndarray): Feature vector of node vi.
    - center_features (np.ndarray): Feature vector of the center node c.
    - m (int): Hop distance of vi from the center node.
    - neighborhood_features (np.ndarray): Feature matrix of all nodes in the M-hop neighborhood.

    Returns:
    - float: Probability of selecting node vi.
    """
    # Compute numerator: exp(-m ||xi - xc||^2)
    numerator = np.exp(-m * np.linalg.norm(vi_features - center_features, ord=2) ** 2)

    # Compute denominator: sum(exp(-m ||xj - xc||^2)) for all vj in neighborhood
    denominator = np.sum(np.exp(-m * np.linalg.norm(neighborhood_features - center_features, axis=1) ** 2))

    return numerator / denominator if denominator > 0 else 0.0