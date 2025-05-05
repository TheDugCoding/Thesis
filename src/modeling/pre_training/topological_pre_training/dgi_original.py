import os.path as osp

import torch
from tqdm import tqdm
import os as os

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed  = 'processed'
relative_path_trained_model = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0].to(device, 'x', 'edge_index')

train_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=128,
                              shuffle=True)
test_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=128)


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels)
        ])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([
            torch.nn.PReLU(hidden_channels),
            torch.nn.PReLU(hidden_channels),
            torch.nn.PReLU(hidden_channels)
        ])

    def forward(self, x, edge_index, batch_size):
        for conv, act in zip(self.convs, self.activations):
            x = conv(x, edge_index)
            x = act(x)
        return x[:batch_size]


def corruption(x, edge_index, batch_size):
    return x[torch.randperm(x.size(0))], edge_index, batch_size


model = DeepGraphInfomax(
    hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(epoch):
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch:02d}'):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(batch.x, batch.edge_index,
                                      batch.batch_size)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pos_z.size(0)
        total_examples += pos_z.size(0)

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    zs = []
    for batch in tqdm(test_loader, desc='Evaluating'):
        pos_z, _, _ = model(batch.x, batch.edge_index, batch.batch_size)
        zs.append(pos_z.cpu())
    z = torch.cat(zs, dim=0)
    train_val_mask = data.train_mask | data.val_mask
    acc = model.test(z[train_val_mask], data.y[train_val_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=10000)
    return acc

with open("training_log.txt", "w") as file:
    for epoch in range(1, 31):
        loss = train(epoch)
        log = f"Epoch {epoch:02d}, Loss: {loss:.6f}\n"
        print(log)
        file.write(log)

test_acc = test()
with open("test_accuracy.txt", "w") as file:
    log = f'Test Accuracy: {test_acc:.4f}'
    print(log)
    file.write(log)

torch.save(model.state_dict(), os.path.join(trained_model_path, 'original_dgi.pth'))