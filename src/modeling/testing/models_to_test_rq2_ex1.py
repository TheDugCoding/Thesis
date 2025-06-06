import copy
import os

import torch
import torch.nn as nn
from torch_geometric.graphgym.register import layer_dict
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE, GAT, GIN
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm
from src.modeling.downstream_task.dgi_and_mlp import build_mlp

from src.modeling.final_framework.framework_complex import DGIPlusGNN
from src.modeling.final_framework.framework_simple import DGIAndGNN
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax_only_topological_features import \
    DeepGraphInfomaxWithoutFlexFronts, EncoderWithoutFlexFrontsGraphsage, corruption_without_flex_fronts
from src.modeling.downstream_task.graphsage_and_mlp import GraphsageWithMLP
from src.modeling.downstream_task.dgi_and_mlp import DGIWithMLP
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/testing/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)


# define here the models to test against the framework

def model_list_rq2_ex1(data):
    """
    :param data: the dataset that is used
    :return: a dict containing all the gnns to test against the framework
    """

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS----"""

    train_loader_gnn_model_complex_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[15, 30],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[15, 30],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[15, 30],
        batch_size=64,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1],
                                                  hidden_channels=128, output_channels=128, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_without_flipping_layer.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer = GraphSAGE(
        in_channels=data.num_features + 128,
        hidden_channels=64,
        num_layers=3,
        out_channels=2,
        dropout=0.2599246924621209,
        act='relu',
        aggr='mean'
    )

    gnn_model_complex_framework_without_front_flex = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework_without_front_flex = torch.optim.Adam(
        gnn_model_complex_framework_without_front_flex.parameters(),
        lr=0.0003481550881584628, weight_decay=2.3566177347305847e-06)
    criterion_gnn_complex_framework_without_front_flex = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS FIRST LAYER UNFREEZE----"""

    train_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[20, 20],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[20, 20],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[20, 20],
        batch_size=64,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1],
                                                  hidden_channels=128, output_channels=128, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    skip_first_layer = True
    for idx, layer in enumerate(dgi_model_without_flipping_layer.encoder.layers):
        if idx == 0 and skip_first_layer:
            continue  # Skip the first layer
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer = GraphSAGE(
        in_channels=data.num_features + 128,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
        dropout=0.22850289637244808,
        act='leaky_relu',
        aggr='sum'
    )

    gnn_model_complex_framework_without_front_flex_first_layer_not_frozen = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                                       gnn_model_downstream_framework_without_flipping_layer,
                                                                                       False)
    optimizer_gnn_complex_framework_without_front_flex_first_layer_not_frozen = torch.optim.Adam(
        gnn_model_complex_framework_without_front_flex_first_layer_not_frozen.parameters(),
        lr=0.0001556396670559418, weight_decay=5.7752955996249056e-06)
    criterion_gnn_complex_framework_without_front_flex_first_layer_not_frozen = torch.nn.CrossEntropyLoss(
        ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS LAST LAYER UNFREEZE----"""

    train_loader_gnn_model_complex_framework_without_front_flex_last_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_without_front_flex_last_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_without_front_flex_last_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=64,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1],
                                                  hidden_channels=128, output_channels=128, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    # freeze all the layers except the last one
    for idx in range(len(dgi_model_without_flipping_layer.encoder.layers) - 1):
        for param in dgi_model_without_flipping_layer.encoder.layers[idx].parameters():
            param.requires_grad = False

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer = GraphSAGE(
        in_channels=data.num_features + 128,
        hidden_channels=256,
        num_layers=4,
        out_channels=2,
        dropout=0.40460730426191116,
        act='gelu',
        aggr='max',
        norm=BatchNorm(256)
    )

    gnn_model_complex_framework_without_front_flex_last_layer_not_frozen = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                                       gnn_model_downstream_framework_without_flipping_layer,
                                                                                       False)
    optimizer_gnn_complex_framework_without_front_flex_last_layer_not_frozen = torch.optim.Adam(
        gnn_model_complex_framework_without_front_flex_last_layer_not_frozen.parameters(),
        lr=0.005370050867649166, weight_decay=2.1789005906367674e-06)
    criterion_gnn_complex_framework_without_front_flex_last_layer_not_frozen = torch.nn.CrossEntropyLoss(
        ignore_index=-1)
    
    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        'complex_framework_without_flex_fronts': {
            'model': gnn_model_complex_framework_without_front_flex,
            'optimizer': optimizer_gnn_complex_framework_without_front_flex,
            'criterion': criterion_gnn_complex_framework_without_front_flex,
            'train_set': train_loader_gnn_model_complex_framework_without_front_flex,
            'val_set': val_loader_gnn_model_complex_framework_without_front_flex,
            'test_set': test_loader_gnn_model_complex_framework_without_front_flex
        },

        'complex_framework_without_flex_fronts_first_layer_not_frozen': {
            'model': gnn_model_complex_framework_without_front_flex_first_layer_not_frozen,
            'optimizer': optimizer_gnn_complex_framework_without_front_flex_first_layer_not_frozen,
            'criterion': criterion_gnn_complex_framework_without_front_flex_first_layer_not_frozen,
            'train_set': train_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen,
            'val_set': val_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen,
            'test_set': test_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen
        },

        'complex_framework_without_flex_fronts_last_layer_not_frozen': {
            'model': gnn_model_complex_framework_without_front_flex_last_layer_not_frozen,
            'optimizer': optimizer_gnn_complex_framework_without_front_flex_last_layer_not_frozen,
            'criterion': criterion_gnn_complex_framework_without_front_flex_last_layer_not_frozen,
            'train_set': train_loader_gnn_model_complex_framework_without_front_flex_last_layer_not_frozen,
            'val_set': val_loader_gnn_model_complex_framework_without_front_flex_last_layer_not_frozen,
            'test_set': test_loader_gnn_model_complex_framework_without_front_flex_last_layer_not_frozen
        }
    }

    return model_dict