import os as os

import torch
from click.core import batch
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import DeepGraphInfomax, SAGEConv, GraphSAGE
from torch_geometric.data import Batch
from src.Scripts.Data_Preparation.preprocess import FinancialGraphDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path_processed  = '../../../../../Data/Processed/'
processed_data_location = os.path.join(script_dir, relative_path_processed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# code taken from this site
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_inductive.py

'''Original loader I will change it because I have multiple graphs 
train_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=256,
                              shuffle=True, num_workers=12)
test_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=256,
                             num_workers=12)

'''






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

def train(epoch):
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch:02d}'):
        batch = batch.to(device)
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(batch.x, batch.edge_index,
                                      batch.batch_size)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pos_z.size(0)
        total_examples += pos_z.size(0)

    return total_loss / total_examples

'''
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

'''

if __name__ == '__main__':

    dataset = FinancialGraphDataset(root=processed_data_location)
    data_list = [dataset[i] for i in range(len(dataset))]

    batch = Batch.from_data_list(dataset)

    train_loader = NeighborLoader(
        batch,
        num_neighbors=[10, 10, 25],  # Sample up to 10 neighbors per layer
        batch_size=256,  # Number of seed nodes per batch
        shuffle=True,
        num_workers=12
    )

    model = DeepGraphInfomax(
        hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, 31):
        loss = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

    torch.save(model.state_dict(), 'modeling_graphsage_unsup_trained.pth')



#test_acc = test()
#print(f'Test Accuracy: {test_acc:.4f}')
