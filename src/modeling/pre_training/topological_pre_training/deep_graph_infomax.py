import os as os

import torch
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
import torch.nn.functional as F
from src.data_preprocessing.preprocess import RealDataTraining
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed  = 'processed'
relative_path_trained_model = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)

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
    def __init__(self, hidden_channels, output_channels, unique_layers):
        super().__init__()

        # First layer, the first layer is unique for each dataset, this way we
        # can solve the problem of multi dimensionality
        for data_conv in range(unique_layers):
            self.dataset_convs.append(SAGEConv(-1, hidden_channels))
        # the second layer is in common with all
        self.conv2 = SAGEConv(hidden_channels, output_channels)


    def forward(self, x, edge_index, batch_size, layer):
        x = F.relu(self.dataset_convs[layer](x, edge_index))
        x = self.conv2(x, edge_index)
        return x[:batch_size]


def corruption(x, edge_index, batch_size):
    return x[torch.randperm(x.size(0))], edge_index, batch_size

def train(epoch, train_loader):
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

    dataset = RealDataTraining(root = processed_data_path, add_topological_features=False)

    data_rabo = dataset[0]
    data_ethereum = dataset[1]

    train_loader = NeighborLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_neighbors=[10, 10],
    )

    model = DeepGraphInfomax(
        hidden_channels=512, encoder=Encoder(64, 64, 2),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, 31):
        loss = train(epoch, train_loader)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

    torch.save(model.state_dict(), os.path.join(trained_model_path, 'modeling_graphsage_unsup_trained.pth'))



#test_acc = test()
#print(f'Test Accuracy: {test_acc:.4f}')
