import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import json
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx


def inductive_node_2_vec(G):
    G = G.to_undirected()
    data = from_networkx(G)

    # Device configuration (use GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create Node2Vec model
    model = Node2Vec(
        data.edge_index,  # The edge_index from the graph
        embedding_dim=128,  # Dimensionality of the embeddings
        walk_length=20,  # Length of each random walk
        context_size=10,  # Context size (like window size in Word2Vec)
        walks_per_node=10,  # Number of walks per node
        num_negative_samples=1,  # Number of negative samples
        p=1.0,  # Return parameter (controls depth of walk)
        q=1.0,  # In-out parameter (controls breadth of walk)
        sparse=True  # Sparse updates for efficiency
    ).to(device)

    data.edge_index = data.edge_index.to(device)

    # Data loader (for batching)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)

    # Optimizer
    optimizer = optim.SparseAdam(list(model.parameters()), lr=0.01)

    # Training loop
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))  # Compute loss for this batch
            loss.backward()
            optimizer.step()  # Update parameters
            total_loss += loss.item()
        return total_loss / len(loader)

    # Training the model for 100 epochs
    for epoch in range(1, 101):
        loss = train()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

    @torch.no_grad()
    def get_embeddings():
        model.eval()
        z = model()
        return z.cpu()

    embeddings = get_embeddings()

    for i, node in enumerate(G.nodes()):
        embedding_tensor = torch.tensor(embeddings[i].numpy())  # Convert to tensor
        G.nodes[node]['deepwalk_embedding'] = embedding_tensor.numpy().tolist()

    return G


def get_structural_info(G):
    '''
    This function compute several structural node measurements, if a ne
    :param G: A Graph in networkX
    :return: A graph G, containing additional structural information
    '''
    nx.set_node_attributes(G, dict(nx.degree(G)), 'degree')
    nx.set_node_attributes(G, nx.degree_centrality(G), 'degree_centrality')

    # Calculate PageRank scores using networkx
    pagerank_scores = nx.pagerank(G, alpha=0.85)

    # Find dangling nodes (nodes with no outgoing edges)
    dangling_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]

    # Calculate the lower bound for PageRank scores
    # rlow = (epsilon + (1 - epsilon) * sum(r(d))) / |V|
    epsilon = 0.15
    dangling_contrib = sum(pagerank_scores[d] for d in dangling_nodes)
    rlow = (epsilon + (1 - epsilon) * dangling_contrib) / len(G.nodes)

    # Normalize the PageRank scores
    normalized_pagerank = {node: score / rlow for node, score in pagerank_scores.items()}

    nx.set_node_attributes(G, normalized_pagerank, 'pagerank_normalized')

    # Normalized Eigenvector Centrality
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    max_ec = max(eigenvector.values())
    ec_norm = {node: val / max_ec for node, val in eigenvector.items()}
    nx.set_node_attributes(G, ec_norm, 'eigenvector_centrality_norm')

    # Clustering Coefficient
    clustering = nx.clustering(G)
    nx.set_node_attributes(G, clustering, 'clustering_coef')

    # inductive deep walk
    return inductive_node_2_vec(G)


def select_nodes(preprocessed_dataset: nx.DiGraph, percentage_of_node_to_sample_per_dataset=100,
                 select_random_nodes=True):
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
        G.nodes[node].get("pagerank", 0),  # Get pagerank, default 0 if missing
        G.nodes[node].get("degree_centrality", 0),  # Get degree centrality
        G.nodes[node].get("degree", 0)  # Get degree
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
