#IMPORT STATEMENTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

df = pd.read_csv("Data/AML_world/LI-Small_Trans.csv")

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