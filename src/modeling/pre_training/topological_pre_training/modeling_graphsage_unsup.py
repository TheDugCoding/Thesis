import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from torch_geometric.nn import SAGEConv, DeepGraphInfomax
from sklearn.manifold import TSNE
from src.data_preparation.preprocess import FinancialGraphDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path_processed  = '../../../data/processed/'
processed_data_location = os.path.join(script_dir, relative_path_processed)



class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)

# ------------------------------ UNSUPERVISED TRAINING WITH DGI ------------------------------

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv = GraphSAGE(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

    def encode(self, x, edge_index):
        return self.forward(x, edge_index)

# corruption function for Deep Graph Infomax
def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index  # Shuffle node features

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
dataset = FinancialGraphDataset(root=processed_data_location)
graphs = [dataset[i] for i in range(len(dataset))]

model = DeepGraphInfomax(
    hidden_channels=64,
    encoder=Encoder(in_channels=3, hidden_channels=64, out_channels=64),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


'''--- Training Function ---'''


def train(graph):
    model.train()
    optimizer.zero_grad()

    # Ensure we only use node features, ignoring edge labels (if any exist)
    pos_z, neg_z, summary = model(graph.x.to(device),
                                  graph.edge_index.to(device))  # edge_index is only for message passing

    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()

    return loss.item(), pos_z


# Train on all graphs
for epoch in range(50):  # Adjust epochs as needed
    total_loss = 0
    for graph in graphs:
        loss, embeddings = train(graph)
        total_loss += loss

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, "modeling_graphsage_unsup_trained.pth")

'''--- t-SNE Visualization ---'''


def visualize_tsne(embeddings):
    num_samples = embeddings.shape[0]
    perplexity = min(30, num_samples - 1)  # Avoid TSNE error

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().detach().numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, c="blue")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.title("TSNE Visualization of Node Embeddings")
    plt.savefig('../../../data/')


def compute_correlation(graphs, embeddings, save_path):
    """
    Compute correlation between initial node features (data.x) and final embeddings.

    :param graph: PyG data object (contains x: initial node features)
    :param embeddings: Learned node embeddings from GraphSAGE
    """
    idx = 0
    for graph in graphs:
        x_features = graph.x.cpu().detach().numpy()  # Initial node features
        z_embeddings = embeddings.cpu().detach().numpy()  # Final latent representations

        # Compute Pearson and Spearman correlation for each feature
        correlations = {}
        for i in range(x_features.shape[1]):  # Iterate over each feature
            feature_values = x_features[:, i]

            # Compute correlation with each dimension in latent space
            for j in range(z_embeddings.shape[1]):
                embedding_values = z_embeddings[:, j]

                pearson_corr, _ = pearsonr(feature_values, embedding_values)
                spearman_corr, _ = spearmanr(feature_values, embedding_values)

                correlations[f"Feature {i} - Embedding {j}"] = (pearson_corr, spearman_corr)

        # Convert to DataFrame
        corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Pearson', 'Spearman'])
        print(corr_df)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, cmap="coolwarm", annot=True)
        plt.title("Correlation between Initial Features and Learned Embeddings")
        img_path = f"{save_path}_{idx}.png"
        plt.savefig(img_path)
        idx = idx +1


visualize_tsne(embeddings)
compute_correlation(graphs, embeddings, processed_data_location)