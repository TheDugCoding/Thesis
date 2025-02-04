import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from scipy.stats import skew
from matplotlib.transforms import Bbox
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
from networkx.algorithms.community import modularity, greedy_modularity_communities



raw_df = pd.read_csv("/Data/enchanced_aml_world/SAML-D.csv")
"""
print(raw_df.shape)
df = raw_df.sample(n=100000, random_state=1)

print(df.head())

sns.countplot(data=df, x='Is_laundering')

plt.figure(figsize=(25, 6))
sns.countplot(data=df, x='Laundering_type')
plt.show()

print(df.columns)
print(df.info())

class_distribution = df['Is_laundering'].value_counts()

plt.figure(figsize=(10, 6))
plt.pie(class_distribution, labels=['Non-Laundering Transactions', 'Suspicious Transactions'], autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])

plt.title('Class Distribution')
plt.axis('equal')

plt.show();

#checking how many unique values there are here
accounts_combined = pd.concat([df['Sender_account'], df['Receiver_account']], axis=0).nunique()

print(f"Number of accounts (): {accounts_combined}")
"""

# Create a graph using NetworkX
G = nx.DiGraph()  # A directed graph is assumed since transactions have a direction (From Bank -> To Bank)

# Iterate through the DataFrame and add edges with attributes
for _, row in raw_df.iterrows():
    from_bank = row['From Bank']
    to_bank = row['To Bank']

    # Create an edge between the 'From Bank' and 'To Bank' with the attributes
    G.add_edge(from_bank, to_bank,
               amount_received=row['Amount Received'],
               receiving_currency=row['Receiving Currency'],
               amount_paid=row['Amount Paid'],
               payment_currency=row['Payment Currency'],
               payment_format=row['Payment Format'],
               is_laundering=row['Is Laundering'])

# 1. **Network Density**
"""
Network density is a measure of how many connections (edges)
 exist in a network compared to the maximum possible number of connections. It provides an indication of how well connected the network is.
  A higher density means that the network is more tightly connected, while a lower density suggests a sparse or fragmented network.
"""
def network_density(G):
    density = nx.density(G)
    print(f'Network Density: {density}')
    return density

"""
Degree distribution is a fundamental characteristic of a network that describes the frequency 
with which nodes in the network have a given degree (number of connections). In other words, it shows how many
 nodes are connected to a certain number of other nodes. This distribution provides insights into the overall structure of the network,
  such as whether most nodes have a similar degree (indicative of a uniform network) or if there are a few highly connected nodes (hubs) 
  alongside many sparsely connected nodes (indicative of a scale-free network). Degree distribution is often visualized as a histogram or a plot, 
  showing the degrees on the x-axis and their frequencies or normalized percentages on the y-axis. This analysis is 
critical in fields like social networks, biological systems, and financial networks, as it helps identify patterns such as hubs,
 robustness, and potential vulnerabilities.
"""

def degree_distribution_histogram(G, n=10):
    # Get the degree of each node
    degrees = [G.degree(n) for n in G.nodes()]

    # Calculate the range of degrees
    min_degree = min(degrees)
    max_degree = max(degrees)

    # Create n bins (bands)
    bins = np.linspace(min_degree, max_degree, n + 1)  # n + 1 edges for n bands

    # Count the number of nodes in each bin
    histogram, bin_edges = np.histogram(degrees, bins=bins)

    # Normalize the counts to percentages
    total_nodes = len(degrees)
    normalized_counts = (histogram / total_nodes) * 100

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(
        x=bin_edges[:-1],  # Use the left edges of bins for x-axis
        height=normalized_counts,
        width=np.diff(bin_edges),  # Bin width
        align='edge',
        color='blue',
        alpha=0.75
    )
    plt.xlabel('Degree (Binned)')
    plt.ylabel('Percentage of Nodes (%)')
    plt.title(f'Degree Distribution with {n} Bands')
    plt.xticks(bin_edges, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Return histogram data for further analysis
    return histogram, bin_edges

# 2. **Degree Distribution**
def degree_distribution(G):
    # Get the degree of each node
    degrees = [G.degree(n) for n in G.nodes()]

    # Count the frequency of each unique degree
    degree_counts = {}
    for degree in degrees:
        if degree in degree_counts:
            degree_counts[degree] += 1
        else:
            degree_counts[degree] = 1

    # Normalize the counts to 100
    total_nodes = sum(degree_counts.values())
    normalized_counts = {k: (v / total_nodes) * 100 for k, v in degree_counts.items()}

    # Extract the degrees and their normalized counts
    degree_values = list(normalized_counts.keys())
    normalized_values = list(normalized_counts.values())

    # Plot the degree distribution
    plt.figure(figsize=(10, 6))
    plt.bar(degree_values, normalized_values, color='blue', alpha=0.75)
    plt.xlabel('Degree')
    plt.ylabel('Percentage of Nodes (%)')
    plt.title('Degree Distribution')
    plt.xticks(degree_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return normalized_counts

"""
Clustering coefficient is a measure of the tendency of nodes in a network to form tightly knit groups or clusters.
It quantifies how connected a node’s neighbors are to each other, representing the likelihood that two neighbors of a node are also connected.
The clustering coefficient can be calculated for individual nodes (local clustering coefficient) or for the entire network (global clustering coefficient).

"""
# 3. **Clustering Coefficient**
def clustering_coefficient(G):
    clustering_coeffs = nx.clustering(G)
    avg_clustering = np.mean(list(clustering_coeffs.values()))
    print(f'Average Clustering Coefficient: {avg_clustering}')
    return avg_clustering

"""
Network modularity is a measure of the structure of a network that quantifies the extent to which the network can be divided into distinct modules
or communities. A module (or community) is a group of nodes that are more densely connected to each other than to the rest of the network.
Modularity helps assess the strength of community structure within a network.
"""
# 4. **Modularity**
def network_modularity(G):
    # First, detect communities (using the Louvain method, Girvan-Newman, or others)
    communities = list(greedy_modularity_communities(G))
    modularity_value = modularity(G, communities)
    print(f'Modularity: {modularity_value}')
    # Print the number of communities
    num_communities = len(communities)
    print(f'Number of communities: {num_communities}')
    # Calculate the ratio of the number of communities to the number of nodes
    num_nodes = len(G.nodes())
    ratio_communities_nodes = num_communities / num_nodes
    print(f'Number of communities / Number of nodes: {ratio_communities_nodes}')
    return modularity_value, communities


# Network Density
density = network_density(G)
print("density" + str(density))
# Degree Distribution
degree_dist = degree_distribution(G)
degree_distribution_histogram(G, 30)

# Clustering Coefficient
clustering = clustering_coefficient(G)

# Modularity and Communities
modularity_value, communities = network_modularity(G)
