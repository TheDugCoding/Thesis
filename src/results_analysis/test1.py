import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphSAGE


# --- Load data ---
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# --- Settings ---
num_layers = 3  # Number of GNN layers
sizes = [10, 20, 40]  # Neighbor sizes mismatch (only 3 hops)
batch_size = 64

# --- Create NeighborSampler ---
sampler = NeighborSampler(
    data.edge_index,
    sizes=sizes,
    batch_size=batch_size,
    shuffle=True,
    num_nodes=data.num_nodes,
)

# --- Initialize model ---
model = GraphSAGE(dataset.num_features, hidden_channels=128, out_channels=dataset.num_classes, num_layers=num_layers)

# --- Optimizer ---
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# --- Training loop ---
def train():
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in sampler:
        optimizer.zero_grad()
        x = data.x[n_id]
        y = data.y[n_id[:batch_size]]
        out = model(x, adjs)  # adjs is a list of tuples, expected by forward
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        break  # just for demonstration
    print(f"Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()
