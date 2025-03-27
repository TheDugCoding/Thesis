import time

import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv

from src.data_preprocessing.preprocess import RealDataTraining
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/downstream_task/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)

dataset = RealDataTraining(root = processed_data_path, add_topological_features=True)

loader = NeighborLoader(graph, num_neighbors=[num_neighbors], batch_size=1, input_nodes=[node_idx])

data = dataset[0]

train_loader = LinkNeighborLoader(
    data,
    batch_size=256,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_neighbors=[10, 10],
)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')
data = data.to(device, 'x', 'edge_index')

class GNN_PRETRAIN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        # First layer, the first layer is unique for each dataset, this way we
        # can solve the problem of multi dimensionality
        for data_conv in range(3):
            self.dataset_convs.append(SAGEConv(-1 , out_channels))
        #the second layer is in common with all
        self.conv2 = SAGEConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


model = GNN_PRETRAIN(data.num_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index).cpu()

    clf = LogisticRegression()
    clf.fit(out[data.train_mask], data.y[data.train_mask])

    val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    return val_acc, test_acc


times = []
for epoch in range(1, 51):
    start = time.time()
    loss = train()
    val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")