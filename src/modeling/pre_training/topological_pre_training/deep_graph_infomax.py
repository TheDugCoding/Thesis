import copy
import os as os
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.inits import reset, uniform
from tqdm import tqdm

from src.data_preprocessing.preprocess import RealDataTraining, AmlTestDataset
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

EPS = 1e-15


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

def train(epoch, train_loaders):
    '''
    :param epoch: for how many epochs we should train the model
    :param train_loaders: the train loaders of the different datasets to use, the biggest trainloader should be in first position
    :return: loss
    '''
    model.train()
    batch_counts = [len(loader) for loader in train_loaders]
    max_batches = batch_counts[0]  # assuming the list is already sorted from largest to smallest

    # handle multi-loader case
    if len(train_loaders) > 1:
        ratios = [round(max_batches / count) for count in batch_counts[1:]]
        iter_list = [iter(loader) for loader in train_loaders[1:]]
    else:
        ratios = []
        iter_list = []

    total_loss = total_examples = 0
    for batch_idx, batch in enumerate(tqdm(train_loaders[0], desc=f'Epoch {epoch:02d}')):

        batches = []

        # we put here the batch of the biggest dataset
        batches.append(batch.to(device))
        # and here we add all the other batches
        for ratio_idx, ratio in enumerate(ratios):
            if batch_idx % ratio == 0:
                try:
                    batches.append(next(iter_list[ratio_idx]).to(device))  # Try getting the next batch
                except StopIteration:
                    iter_list[ratio_idx] = iter(train_loaders[ratio_idx+1])  # Reset iterator when exhausted
                    batches.append(next(iter_list[ratio_idx]).to(device))  # Get next batch

        for idx, batch_loop in enumerate(batches):
            optimizer.zero_grad()
            pos_z, neg_z, summary = model(batch_loop.x, batch_loop.edge_index,
                                          batch_loop.batch_size, layer=idx)
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

    dataset = RealDataTraining(root = processed_data_path, add_topological_features = False)

    data_rabo = dataset[0]
    data_ethereum = dataset[1]

    train_loader_rabo = NeighborLoader(
        data_rabo,
        batch_size=256,
        shuffle=True,
        num_neighbors=[10, 10, 25],
    )

    train_loader_ethereum = NeighborLoader(
        data_ethereum,
        batch_size=256,
        shuffle=True,
        num_neighbors=[10, 10, 25],
    )
    '''
    dataset = AmlTestDataset(root=processed_data_path, add_topological_features=False)

    data = dataset[0]

    train_loader_aml = NeighborLoader(
        data,
        batch_size=256,
        shuffle=True,
        num_neighbors=[10, 10, 25]
    )
    '''

    #set the train loader from the biggest to the smallest, otherwise it won't work
    train_loaders = [train_loader_ethereum]

    #define the model, the unique layers correspond to the number of "flipping layers", meaning that
    #each dataset has its own layer
    model = DeepGraphInfomax(
        hidden_channels=64, encoder=Encoder(64, 64, len(train_loaders)),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    with open("training_log_only_ethereum_topo_false.txt", "w") as file:
        for epoch in range(1, 30):
            loss = train(epoch, train_loaders)
            log = f"Epoch {epoch:02d}, Loss: {loss:.6f}\n"
            print(log)
            file.write(log)

    torch.save(model.state_dict(), os.path.join(trained_model_path, 'modeling_only_ethereum_topo_false.pth'))

# test_acc = test()
# print(f'Test Accuracy: {test_acc:.4f}')