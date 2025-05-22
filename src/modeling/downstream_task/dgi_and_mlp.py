import copy
import os

import torch
import torch.nn as nn
from torch_geometric.graphgym.register import layer_dict
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE, GAT, GIN
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm

from src.modeling.final_framework.framework_complex import DGIPlusGNN
from src.modeling.final_framework.framework_simple import DGIAndGNN
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax_only_topological_features import \
    DeepGraphInfomaxWithoutFlexFronts, EncoderWithoutFlexFrontsGraphsage, corruption_without_flex_fronts
from src.modeling.downstream_task.graphsage_and_mlp import GraphsageWithMLP
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder


def build_mlp(layer_sizes, activation_fn, dropout_rate=0.0):
    """
    Dynamically builds an MLP based on a list of layer sizes.

    Args:
        layer_sizes (List[int]): A list of integers where each element is the size of a layer.
                                 Example: [128, 64, 32, 2] will create:
                                 Linear(128 -> 64) -> ReLU -> Linear(64 -> 32) -> ReLU -> Linear(32 -> 2)

    Returns:
        nn.Sequential: The constructed MLP.
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(activation_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)

class DGIWithMLP(nn.Module):
    def __init__(self, dgi_model: nn.Module,mlp):
        super().__init__()
        self.dgi = dgi_model
        self.mlp = mlp

    def forward(self, batch):
        # it refers to the DGI model and the special "flipping layer"
        topological_latent_representation = self.dgi(batch.topological_features, batch.edge_index,
                                                         batch.batch_size, framework=True)
        return self.mlp(topological_latent_representation[0])