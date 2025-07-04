import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv


# --- Define the GNN model with num_layers and print info in forward ---
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        # adjs is a list of (edge_index, e_id, size) tuples, one per hop
        print(f"Forward pass with {len(adjs)} hops, model layers: {self.num_layers}")
        for i, (edge_index, _, size) in enumerate(adjs):
            if i >= self.num_layers:
                print(f"Skipping extra hop {i} (no corresponding layer)")
                break
            x_target = x[:size[1]]  # Target nodes are always placed first
            print(
                f" Layer {i}: x.shape={x.shape}, x_target.shape={x_target.shape}, edge_index.shape={edge_index.shape}")
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x


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
        print(f"\nBatch with {len(adjs)} hops")
        optimizer.zero_grad()
        x = data.x[n_id]
        y = data.y[n_id[:batch_size]]

        # Forward pass
        out = model(x, adjs)

        # Calculate loss only for the root nodes
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # For demonstration, break after first batch
        break
    print(f"Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()
