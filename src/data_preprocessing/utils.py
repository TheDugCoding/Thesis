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

    for i, node in enumerate(G.nodes):
        embedding_str = ','.join(map(str, embeddings[i].numpy()))  # Convert numpy array to comma-separated string
        G.nodes[node]['deepwalk_embedding'] = embedding_str

    return G


def get_structural_info(G):
    '''
    This function compute several structural node measurements, if a ne
    :param G: A Graph in networkX
    :return: A graph G, containing additional structural information
    '''

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.DiGraph(G) if G.is_directed() else nx.Graph(G)

    nx.set_node_attributes(G, dict(nx.degree(G)), 'degree')

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

    # inductive deep walk, too expensive
    #inductive_node_2_vec(G)

    return G
