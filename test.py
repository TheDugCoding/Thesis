import networkx as nx
import matplotlib.pyplot as plt

# Define the adjacency matrix
adj_matrix = [
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
]

# Create an empty graph
G = nx.Graph()

# Add nodes (assuming nodes are labeled 1 through 4)
nodes = [1, 2, 3, 4]
G.add_nodes_from(nodes)

# Add edges based on adjacency matrix
for i in range(len(adj_matrix)):
    for j in range(i + 1, len(adj_matrix)):
        if adj_matrix[i][j] == 1:
            G.add_edge(i + 1, j + 1)

# Draw the graph
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=16)
plt.title("Graph from Adjacency Matrix")
plt.show()