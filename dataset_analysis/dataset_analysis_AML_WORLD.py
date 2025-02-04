#IMPORT STATEMENTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.algorithms.community import modularity, greedy_modularity_communities


df = pd.read_csv("../Data/AML_world/LI-Small_Trans.csv")

#features
print(df.columns)
print(df.head(5))
print(df.shape)

#check datset composition
print(df.isna().sum())
print(df.duplicated().sum())
print(df.drop_duplicates(inplace=True))

#payment format in money laundering
df["Payment Format"].unique()
payment_format=df[df["Is Laundering"]==1]["Payment Format"].value_counts()


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.barplot(x=payment_format.index,y=payment_format.values)

plt.subplot(1,2,2)
plt.pie(payment_format.values,labels=payment_format.index,wedgeprops=dict(width=0.4)); #use semicolon to avoid texts in O/P
plt.title("Payment Format Distribution in money laundering")
plt.show()

#payment format in not money laundering

df["Payment Format"].unique()
payment_format=df[df["Is Laundering"]==0]["Payment Format"].value_counts()

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.barplot(x=payment_format.index,y=payment_format.values)

plt.subplot(1,2,2)
plt.pie(payment_format.values,labels=payment_format.index,wedgeprops=dict(width=0.4)); #use semicolon to avoid texts in O/P
plt.title("Payment Format Distribution in clean activities")
plt.show()

#correlation matrix
cr=df.corr(numeric_only=True)
sns.heatmap(cr,annot=True,cmap="Blues")
plt.show()

#show amount of is laundering
print(df["Is Laundering"].value_counts())
sns.countplot(data=df, x='Is Laundering')
plt.show()

#check how many transfers are from different banks
same_bank_count = (df['From Bank'] == df['To Bank']).sum()  # Count same-bank transactions
cross_bank_count = (df['From Bank'] != df['To Bank']).sum()  # Count cross-bank transactions

print(f"Number of same-bank transactions: {same_bank_count}")
print(f"Number of cross-bank transactions: {cross_bank_count}")

# Plot the counts
plt.figure(figsize=(15, 8))
counts = pd.Series({'Same Bank': same_bank_count, 'Cross Bank': cross_bank_count})
counts.plot(kind='bar', color=['green', 'red'], title='Same Bank vs Cross Bank Transactions')
plt.ylabel('Number of Transactions')
plt.title("NON ML transfer type")
plt.show()

#check how many ML transfers are from different banks
laundering_df = df[df['Is Laundering'] == 1]
same_bank_count = (laundering_df['From Bank'] == laundering_df['To Bank']).sum()  # Count same-bank transactions
cross_bank_count = (laundering_df['From Bank'] != laundering_df['To Bank']).sum()  # Count cross-bank transactions

print(f"Number of same-bank transactions: {same_bank_count}")
print(f"Number of cross-bank transactions: {cross_bank_count}")

# Plot the counts
plt.figure(figsize=(15, 8))
counts = pd.Series({'Same Bank': same_bank_count, 'Cross Bank': cross_bank_count})
counts.plot(kind='bar', color=['green', 'red'], title='Same Bank vs Cross Bank Transactions')
plt.ylabel('Number of Transactions')
plt.title("ML transfer type")
plt.show()

#find the components of the graph
# Step 1: Filter rows where Is Laundering is 1
laundering_df = df[df['Is Laundering'] == 1]

# Step 2: Create a graph from the filtered DataFrame
G = nx.Graph()
G.add_edges_from(zip(laundering_df['From Bank'], laundering_df['To Bank']))

# Step 3: Find connected components
num_components = nx.number_connected_components(G)
print(f"Number of connected components: {num_components}")

# Step 4: Find connected components, calculate node count, edge count, density, and highest degree centrality
components = list(nx.connected_components(G))
for i, component in enumerate(components, start=1):
    subgraph = G.subgraph(component)  # Create subgraph for the component
    num_nodes = len(component)
    num_edges = subgraph.number_of_edges()

    # Calculate density
    if num_nodes > 1:  # Avoid division by zero for single-node components
        density = 2 * num_edges / (num_nodes * (num_nodes - 1))
    else:
        density = 0  # A single node has no edges, so density is 0

    # Calculate degree centrality for each node
    degree_centrality = nx.degree_centrality(subgraph)

    # Find the node with the highest degree centrality
    highest_centrality_node = max(degree_centrality, key=degree_centrality.get)
    highest_centrality_value = degree_centrality[highest_centrality_node]

    print(f"Component {i}: {component} - Number of nodes: {num_nodes}, Number of edges: {num_edges}, "
          f"Density: {density:.4f}")
    print(
        f"  Node with highest degree centrality: {highest_centrality_node} - Centrality: {highest_centrality_value:.4f}")

# Step 4: Assign a color to each component
color_map = {}
for i, component in enumerate(components):
    for node in component:
        color_map[node] = i  # Map each node to a component index

# Assign colors to nodes based on their component
node_colors = [color_map[node] for node in G.nodes()]

# Step 5: Plot the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # Layout for better visualization
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,
    cmap=plt.cm.tab10,  # Color map for distinct component colors
    node_size=500,
    font_size=10,
    font_color="white",
    edge_color="gray",
)
plt.title("Connected Components of Laundering Transactions", fontsize=16)
plt.show()


# Count occurrences for 'From Bank' in laundering cases
from_bank_counts = df[df["Is Laundering"] == 1]["From Bank"].value_counts()

# Count occurrences for 'To Bank' in laundering cases
to_bank_counts = df[df["Is Laundering"] == 1]["To Bank"].value_counts()

# Plot for 'From Bank'
plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
sns.barplot(x=from_bank_counts.index, y=from_bank_counts.values)
plt.title("From Bank Distribution in Money Laundering")
plt.xticks(rotation=45)

plt.subplot(2,2,2)
plt.pie(from_bank_counts.values, labels=from_bank_counts.index, wedgeprops=dict(width=0.4))
plt.title("From Bank Distribution Proportion")

# Plot for 'To Bank'
plt.subplot(2,2,3)
sns.barplot(x=to_bank_counts.index, y=to_bank_counts.values)
plt.title("To Bank Distribution in Money Laundering")
plt.xticks(rotation=45)

plt.subplot(2,2,4)
plt.pie(to_bank_counts.values, labels=to_bank_counts.index, wedgeprops=dict(width=0.4))
plt.title("To Bank Distribution Proportion")

plt.tight_layout()
plt.show()

#BANK

# Count occurrences for 'From Bank' in laundering cases
from_bank_counts = df[df["Is Laundering"] == 1]["From Bank"].value_counts()

# Count occurrences for 'To Bank' in laundering cases
to_bank_counts = df[df["Is Laundering"] == 1]["To Bank"].value_counts()

# Plot for 'From Bank'
plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
sns.barplot(x=from_bank_counts.index, y=from_bank_counts.values)
plt.title("From Bank Distribution in Money Laundering")
plt.xticks(rotation=45)

plt.subplot(2,2,2)
plt.pie(from_bank_counts.values, labels=from_bank_counts.index, wedgeprops=dict(width=0.4))
plt.title("From Bank laundering Distribution Proportion")

# Plot for 'To Bank'
plt.subplot(2,2,3)
sns.barplot(x=to_bank_counts.index, y=to_bank_counts.values)
plt.title("To Bank Distribution in Money Laundering")
plt.xticks(rotation=45)

plt.subplot(2,2,4)
plt.pie(to_bank_counts.values, labels=to_bank_counts.index, wedgeprops=dict(width=0.4))
plt.title("To Bank Laundering Distribution Proportion")

plt.tight_layout()
plt.show()

# Get unique banks from 'From Bank' and 'To Bank'
unique_banks = pd.concat([df["From Bank"], df["To Bank"]]).nunique()

# Print the total number of unique banks
print(f"Number of banks: {unique_banks}")

# Get the unique accounts from the 'Account' column
unique_accounts = df["Account"].unique()
# Print the total number of unique banks
print(f"Number of banks: {unique_accounts}")

# Create a graph using NetworkX
G = nx.DiGraph()  # A directed graph is assumed since transactions have a direction (From Bank -> To Bank)

# Iterate through the DataFrame and add edges with attributes
for _, row in df.iterrows():
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
# Degree Distribution
degree_dist = degree_distribution(G)
degree_distribution_histogram(G, 30)

# Clustering Coefficient
clustering = clustering_coefficient(G)

# Modularity and Communities
modularity_value, communities = network_modularity(G)

