import copy
import time
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import Module, Parameter
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn.inits import reset, uniform

from src.data_preprocessing.preprocess import RealDataTraining
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/downstream_task/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)

dataset = RealDataTraining(root = processed_data_path, add_topological_features=False)

EPS = 1e-15

class DeepGraphInfomax(torch.nn.Module):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (torch.nn.Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """
    def __init__(
        self,
        hidden_channels: int,
        encoder: Module,
        summary: Callable,
        corruption: Callable,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = Parameter(torch.empty(hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def forward(self, *args, layer, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation.
        """
        pos_z = self.encoder(*args, layer=layer, **kwargs)

        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        cor_args = cor[:len(args)]
        cor_kwargs = copy.copy(kwargs)
        for key, value in zip(kwargs.keys(), cor[len(args):]):
            cor_kwargs[key] = value

        neg_z = self.encoder(*cor_args, layer=layer, **cor_kwargs)

        summary = self.summary(pos_z, *args, **kwargs)

        return pos_z, neg_z, summary

    def discriminate(self, z: Tensor, summary: Tensor,
                     sigmoid: bool = True) -> Tensor:
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (torch.Tensor): The latent space.
            summary (torch.Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z: Tensor, neg_z: Tensor, summary: Tensor) -> Tensor:
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 -
                              self.discriminate(neg_z, summary, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = 'lbfgs',
        *args,
        **kwargs,
    ) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hidden_channels})'

class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, output_channels, unique_layers):
        super().__init__()

        self.dataset_convs = torch.nn.ModuleList()

        # First layer, the first layer is unique for each dataset, this way we
        # can solve the problem of multi dimensionality
        for data_conv in range(unique_layers):
            self.dataset_convs.append(SAGEConv(-1, hidden_channels))
        # the second layer is in common with all
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, output_channels)




    def forward(self, x, edge_index, batch_size, layer):
        act = torch.nn.PReLU().to(device)
        x = act(self.dataset_convs[layer](x, edge_index))
        x = act(self.conv2(x, edge_index))
        x = act(self.conv3(x, edge_index))
        return x[:batch_size]


def corruption(x, edge_index, batch_size):
    return x[torch.randperm(x.size(0))], edge_index, batch_size

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)
        return x

class DGIPlusGNN(torch.nn.Module):
    def __init__(self, dgi, classifier):
        super().__init__()
        self.dgi = dgi
        self.classifier = classifier

    def forward(self, x, edge_index):
        x = self.dgi(x, edge_index)


        return x



#loader = NeighborLoader(graph, num_neighbors=[num_neighbors], batch_size=1, input_nodes=[node_idx])

data_rabo = dataset[0]
data_ethereum = dataset[1]

train_loader_rabo = LinkNeighborLoader(
    data_rabo,
    batch_size=256,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_neighbors=[10, 10],
)
train_loader_ethereum = LinkNeighborLoader(
    data_ethereum,
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
data_rabo = data_rabo.to(device, 'x', 'edge_index')
data_ethereum = data_ethereum.to(device, 'x', 'edge_index')

#define the framework, first DGI and then the GNN used in the downstream task
dgi_model = DeepGraphInfomax(
        hidden_channels=64, encoder=Encoder(64, 64, 2),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
#load the pretrained parameters
dgi_model.load_state_dict(torch.load("model.pth"))
#freeze layers 2 and 3, let layer 1 learn
dgi_model.conv2.weight.requires_grad = False
dgi_model.conv2.bias.requires_grad = False
dgi_model.conv3.weight.requires_grad = False
dgi_model.conv3.bias.requires_grad = False


gnn_model = GAT(data.num_features, 64, 3,
            8).to(device)
model = DGIPlusGNN(dgi_model, gnn_model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():

    model.train()

    total_loss = 0
    for batch in train_loader_rabo:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index, layer=0)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    for batch in train_loader_ethereum:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index, layer=1)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / (data_ethereum.num_nodes + data_rabo.num_node)

'''
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index).cpu()

    clf = LogisticRegression()
    clf.fit(out[data.train_mask], data.y[data.train_mask])

    val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    return val_acc, test_acc
'''

times = []
for epoch in range(1, 51):
    start = time.time()
    loss = train()
    #val_acc, test_acc = test()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
     #     f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")