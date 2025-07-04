import copy
import os as os
from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import Sequential, Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GINConv
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.utils import erdos_renyi_graph
from tqdm import tqdm

from src.data_preprocessing.preprocess import RealDataTraining
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

EPS = 1e-15

script_dir = get_data_folder()
relative_path_processed = 'processed'
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


class DeepGraphInfomaxWithoutFlexFronts(torch.nn.Module):
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

    def forward(self, *args, framework, num_negatives=1, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns the latent space for the input arguments, their
        corruptions and their summary representation.
        if used with infonce loss increase the number of num negatives
        Returns:
            - pos_z: latent embeddings of input
            - neg_z: corrupted embeddings (1 or more sets)
            - summary: summary vector of pos_z
        """
        # positive embeddings
        # framework: if DGI is part of the framework then True, otherwise False
        pos_z = self.encoder(*args, framework=framework, **kwargs)

        # Compute negatives
        if num_negatives == 1:
            # Standard single negative sample
            cor = self.corruption(*args, **kwargs)
            cor = cor if isinstance(cor, tuple) else (cor,)
            cor_args = cor[:len(args)]
            cor_kwargs = copy.copy(kwargs)
            for key, value in zip(kwargs.keys(), cor[len(args):]):
                cor_kwargs[key] = value

            neg_z = self.encoder(*cor_args, framework=framework, **cor_kwargs)

        else:
            # Multiple negative samples
            neg_z_list = []
            for _ in range(num_negatives):
                cor = self.corruption(*args, **kwargs)
                cor = cor if isinstance(cor, tuple) else (cor,)
                cor_args = cor[:len(args)]
                cor_kwargs = copy.copy(kwargs)
                for key, value in zip(kwargs.keys(), cor[len(args):]):
                    cor_kwargs[key] = value

                neg = self.encoder(*cor_args, framework=framework, **cor_kwargs)
                neg_z_list.append(neg)

            # Stack negatives into a single tensor of shape [num_negatives, N, D]
            neg_z = torch.stack(neg_z_list, dim=0)

        # Summary from positive embeddings
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

    def loss_info_nce(self, pos_z: Tensor, neg_z: Tensor, summary: Tensor, temperature=0.07) -> Tensor:
        """
        pos_z: [N, D]  - positive (real) node embeddings
        neg_z: [N, D]  - negative (corrupted) node embeddings
        summary: [D] or [1, D] - global summary vector
        """

        # Positive logits (one per sample)
        pos_logits = self.discriminate(pos_z, summary, sigmoid=False)

        # Negative logits: each sample's pos summary vs all negative z
        # summary: [N, D], neg_z: [N, D] → need pairwise score
        neg_logits = []
        for k in range(neg_z.size(0)):  # Loop over K
            logits_k = self.discriminate(neg_z[k], summary, sigmoid=False)  # [N]
            neg_logits.append(logits_k)

        neg_logits = torch.stack(neg_logits, dim=0)


        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits.permute(1, 0)], dim=1)  # [N, 1 + N]
        logits = logits / temperature

        # labels: 0 is the correct positive index in each row
        labels = torch.zeros(pos_z.size(0), dtype=torch.long, device=pos_z.device)

        # InfoNCE loss
        loss = torch.nn.CrossEntropyLoss()
        return loss(logits, labels)

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


class EncoderWithoutFlexFrontsGraphsage(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, layers, activation_fn=nn.ReLU):
        """
        :param input_channels: (int) Number of input features per node.
        :param hidden_channels: (int) Number of hidden units in each GAT layer (except the final layer).
        :param output_channels: (int) Number of output features per node from the final GAT layer.
        :param n_layers: (int) Total number of Graphsage layers in the encoder. Must be >= 2.
        :param activation_fn: (Callable) Activation function class to apply after each Graphsage layer (e.g., nn.ReLU, nn.LeakyReLU).
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # First layer
        self.layers.append(SAGEConv(input_channels, hidden_channels))
        self.activations.append(activation_fn())

        # Hidden layers
        for _ in range(layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
            self.activations.append(activation_fn())

        # Final layer
        self.layers.append(SAGEConv(hidden_channels, output_channels))
        self.activations.append(activation_fn())  # Optional: apply activation to output layer

    def forward(self, x, edge_index, batch_size, framework):
        for conv, act in zip(self.layers, self.activations):
            x = act(conv(x, edge_index))

        if framework:
            return x
        else:
            return x[:batch_size]

class EncoderWithoutFlexFrontsGIN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, layers, activation_fn=nn.ReLU):
        """
        :param input_channels: (int) Number of input features per node.
        :param hidden_channels: (int) Hidden layer size for each MLP inside GINConv.
        :param output_channels: (int) Final output feature dimension.
        :param layers: (int) Number of GINConv layers.
        :param activation_fn: (Callable) Activation function class to use (e.g., nn.ReLU).
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # First layer
        mlp = Sequential(
            Linear(input_channels, hidden_channels),
            activation_fn(),
            Linear(hidden_channels, hidden_channels)
        )
        self.layers.append(GINConv(mlp))
        self.activations.append(activation_fn())

        # Hidden layers
        for _ in range(layers - 2):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                activation_fn(),
                Linear(hidden_channels, hidden_channels)
            )
            self.layers.append(GINConv(mlp))
            self.activations.append(activation_fn())

        # Final layer
        mlp = Sequential(
            Linear(hidden_channels, output_channels),
            activation_fn(),
            Linear(output_channels, output_channels)
        )
        self.layers.append(GINConv(mlp))
        self.activations.append(activation_fn())

    def forward(self, x, edge_index, batch_size, framework):
        for conv, act in zip(self.layers, self.activations):
            x = act(conv(x, edge_index))

        if framework:
            return x
        else:
            return x[:batch_size]

class EncoderWithoutFlexFrontsGAT(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, n_layers, activation_fn=nn.ReLU):
        """
        :param input_channels: (int) Number of input features per node.
        :param hidden_channels: (int) Number of hidden units in each GAT layer (except the final layer).
        :param output_channels: (int) Number of output features per node from the final GAT layer.
        :param n_layers: (int) Total number of GAT layers in the encoder. Must be >= 2.
        :param activation_fn: (Callable) Activation function class to apply after each GAT layer (e.g., nn.ReLU, nn.LeakyReLU).
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # First layer
        self.layers.append(GATConv(input_channels, hidden_channels))
        self.activations.append(activation_fn())

        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GATConv(hidden_channels, hidden_channels))
            self.activations.append(activation_fn())

        # Final layer
        self.layers.append(GATConv(hidden_channels, output_channels))
        self.activations.append(activation_fn())

    def forward(self, x, edge_index, batch_size, framework):
        for conv, act in zip(self.layers, self.activations):
            x = act(conv(x, edge_index))

        if framework:
            return x
        else:
            return x[:batch_size]

def corruption_without_flex_fronts(x, edge_index, batch_size):
    return x[torch.randperm(x.size(0))], edge_index, batch_size

def corruption_without_flex_fronts_random_graph_corruptor(x, edge_index, batch_size=None, edge_prob=0.01):
    """
    Creates a random graph using Erdos-Renyi model as a corrupted view for DGI.

    Args:
        x (Tensor): Original node features [N, F]
        edge_index (LongTensor): Original edge index [2, E] (not used here)
        batch_size: Optional, returned unchanged
        edge_prob (float): Probability of edge creation between nodes

    Returns:
        x_corrupt (Tensor): Randomly permuted node features [N, F]
        edge_index_random (LongTensor): Random edge index [2, E']
        batch_size (unchanged)
    """
    num_nodes = x.size(0)
    device = x.device

    # Optionally, randomly shuffle node features
    x_corrupt = x[torch.randperm(x.size(0))]

    # Generate random edges (Erdős-Rényi graph)
    edge_index_random = erdos_renyi_graph(num_nodes, edge_prob)
    edge_index_random = edge_index_random.to(x.device)

    return x_corrupt, edge_index_random, batch_size


def train(epoch, train_loaders, model, optimizer, loss_fun_name):
    '''
    :param epoch: for how many epochs we should train the model
    :param train_loaders: the train loaders of the different datasets to use, the biggest trainloader should be in first position
    :return: loss
    '''
    if loss_fun_name not in ['infoNCE', 'BCEdgi']:
        raise ValueError(f"Invalid loss function name: {loss_fun_name}. Supported values are 'infoNCE' and 'BCEdgi'.")
    #order the loaders from biggest to smallest
    train_loaders = sorted(train_loaders, key=len, reverse=True)
    model.train()
    batch_counts = [len(loader) for loader in train_loaders]
    max_batches = batch_counts[0]

    # handle multi-loader case
    if len(train_loaders) > 1:
        ratios = [round(max_batches / count) for count in batch_counts[1:]]
        iter_list = [iter(loader) for loader in train_loaders[1:]]
    else:
        ratios = []
        iter_list = []

    total_loss = total_examples = 0
    for batch_idx, batch in enumerate(
            tqdm(train_loaders[0], desc=f'Epoch {epoch:02d}', mininterval=20.0)
    ):

        batches = []

        # we put here the batch of the biggest dataset
        batches.append(batch.to(device))
        # and here we add all the other batches
        for ratio_idx, ratio in enumerate(ratios):
            if batch_idx % ratio == 0:
                try:
                    batches.append(next(iter_list[ratio_idx]).to(device))  # Try getting the next batch
                except StopIteration:
                    iter_list[ratio_idx] = iter(train_loaders[ratio_idx + 1])  # Reset iterator when exhausted, we use +1 because the ratio_idx range from 0 to len(train_loaders)-1
                    batches.append(next(iter_list[ratio_idx]).to(device))  # Get next batch

        for idx, batch_loop in enumerate(batches):
            optimizer.zero_grad()
            #if used in combination with info NCE loss change the number on negatives examples
            if loss_fun_name == 'infoNCE':
                pos_z, neg_z, summary = model(batch_loop.x, batch_loop.edge_index,
                                              batch_loop.batch_size, framework=False, num_negatives=3)
                loss = model.loss_info_nce(pos_z, neg_z, summary)
            else:
                #dgi loss
                pos_z, neg_z, summary = model(batch_loop.x, batch_loop.edge_index,
                                              batch_loop.batch_size, framework=False, num_negatives=1)
                loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pos_z.size(0)
            total_examples += pos_z.size(0)

    return total_loss / total_examples


if __name__ == '__main__':

    dataset = RealDataTraining(root = processed_data_path)

    data_rabo = dataset[0]
    data_ethereum = dataset[1]
    data_stable_20 = dataset[2]

    # x contains a dummy feature, replace it with only topological features (only degree for this run)
    data_rabo.x = data_rabo.topological_features[:, 0].unsqueeze(-1)
    data_ethereum.x = data_ethereum.topological_features[:, 0].unsqueeze(-1)
    data_stable_20.x = data_stable_20.topological_features[:, 0].unsqueeze(-1)
    train_loader_rabo = NeighborLoader(
        data_rabo,
        batch_size=64,
        shuffle=True,
        num_neighbors=[10, 10, 25]
    )

    train_loader_ethereum = NeighborLoader(
        data_ethereum,
        batch_size=64,
        shuffle=True,
        num_neighbors=[10, 10, 25]
    )

    train_loader_stable_20 = NeighborLoader(
        data_stable_20,
        batch_size=64,
        shuffle=True,
        num_neighbors=[10, 10, 25]
    )

    # set the train loaders
    train_loaders = [train_loader_rabo, train_loader_ethereum, train_loader_stable_20]

    # define the model, no flexfront
    model = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=32, encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_rabo.num_features, hidden_channels=64, output_channels=32, layers=4, activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0011865761848863178)

    with open("training_log_elliptic_no_flex_front_Graphsage_only_degree.txt", "w") as file:
        for epoch in range(1, 5):
            loss = train(epoch, train_loaders, model, optimizer, 'BCEdgi')
            log = f"Epoch {epoch:02d}, Loss: {loss:.6f}\n"
            print(log)
            file.write(log)

    torch.save(model.state_dict(),
               os.path.join(trained_model_path, 'modeling_dgi_GraphSage_no_flex_front_only_topo_rabo_ecr_20_only_degree.pth'))

# test_acc = test()
# print(f'Test Accuracy: {test_acc:.4f}')