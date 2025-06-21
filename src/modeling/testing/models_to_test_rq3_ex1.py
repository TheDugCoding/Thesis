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
import ast

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/testing/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
relative_path_rq3_finetuning = 'modeling/testing/rq3_ex1_finetuning'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)
models_finetuning_path = get_src_sub_folder(relative_path_rq3_finetuning)

def parse_hyperparams(filepath):
    hyperparams = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()

            # Convert to correct Python type if possible
            try:
                val = ast.literal_eval(val)
            except:
                pass

            hyperparams[key] = val

    return hyperparams

def get_norm(norm_type, hidden_channels):
    if norm_type == "batch":
        return BatchNorm(hidden_channels)
    elif norm_type == "layer":
        return LayerNorm(hidden_channels)
    elif norm_type == "graph":
        return GraphNorm(hidden_channels)
    else:
        return None



def reduce_train_val_masks(data, n_train, n_val):
    """
    Creates two balanced masks (train and val) from the original train_mask,
    with no overlap between them.

    Args:
        data: PyG Data object with .train_mask and .y
        n_train: total number of training samples to keep
        n_val: total number of validation samples to keep

    Returns:
        train_mask_new: Boolean mask with n_train True values, balanced across classes
        val_mask_new: Boolean mask with n_val True values, balanced across classes
    """
    y = data.y
    train_mask = data.train_mask

    unique_classes = torch.unique(y[train_mask])
    num_classes = len(unique_classes)

    train_per_class = n_train // num_classes
    val_per_class = n_val // num_classes

    train_mask_new = torch.zeros_like(train_mask, dtype=torch.bool)
    val_mask_new = torch.zeros_like(train_mask, dtype=torch.bool)

    for c in unique_classes:
        # indices of class c in original train_mask
        idx = (y == c) & train_mask
        class_indices = idx.nonzero(as_tuple=True)[0]

        # shuffle indices
        shuffled = class_indices[torch.randperm(len(class_indices))]

        # split into train and val, no overlap
        train_indices = shuffled[:train_per_class]
        val_indices = shuffled[train_per_class:train_per_class + val_per_class]

        train_mask_new[train_indices] = True
        val_mask_new[val_indices] = True

    return train_mask_new, val_mask_new


# define here the models to test against the framework

def model_list_rq3_ex1(data, n_samples_train):
    """
    :param data: the dataset that is used
    :return: a dict containing all the gnns to test against the framework
    """

    n_samples_val = 100

    train_mask, val_mask = reduce_train_val_masks(data, n_samples_train, n_samples_val)

    activation_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU
    }

    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP----"""

    filepath = os.path.join(models_finetuning_path, f'framework_simple_finetuning_train_set_size_{n_samples_train}.txt')
    hyperparams_simple_framework = parse_hyperparams(filepath)

    train_loader_gnn_model_simple_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_simple_framework.get('neighbours_size'),
        batch_size=64,
        input_nodes=train_mask
    )

    val_loader_gnn_model_simple_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_simple_framework.get('neighbours_size'),
        batch_size=64,
        input_nodes=val_mask
    )

    test_loader_gnn_model_simple_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_simple_framework.get('neighbours_size'),
        batch_size=64,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1],
                                                  hidden_channels=128, output_channels=128, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_simple_framework.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_simple_framework.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in garphsage_elliptic
    gnn_model_downstream_simple_framework = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=hyperparams_simple_framework.get('hidden_channels'),
        num_layers=hyperparams_simple_framework.get('num_layers'),
        out_channels=hyperparams_simple_framework.get('output_channels'),
        dropout=hyperparams_simple_framework.get('dropout'),
        act=hyperparams_simple_framework.get('act'),
        aggr=hyperparams_simple_framework.get('aggr'),
        norm=get_norm(hyperparams_simple_framework.get('norm'), hyperparams_simple_framework.get('hidden_channels'))
    )

    activation_fn = activation_map[hyperparams_simple_framework.get('act_mlp')]

    layer_sizes = [128 + hyperparams_simple_framework.get('output_channels')] + [hyperparams_simple_framework.get('hidden_channels_mlp')] * hyperparams_simple_framework.get('num_layers_mlp') + [2]
    mlp = build_mlp(layer_sizes, activation_fn, hyperparams_simple_framework.get('dropout_mlp'))

    gnn_model_simple_framework_without_front_flex = DGIAndGNN(dgi_model_simple_framework,
                                                              gnn_model_downstream_simple_framework, mlp, False)
    optimizer_gnn_simple_framework_without_front_flex = torch.optim.Adam(
        gnn_model_simple_framework_without_front_flex.parameters(),
        lr=hyperparams_simple_framework.get('lr'), weight_decay=hyperparams_simple_framework.get('weight_decay'))
    criterion_gnn_simple_framework_without_front_flex = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS----"""

    filepath = os.path.join(models_finetuning_path, f'framework_complex_finetuning_train_set_size_{n_samples_train}.txt')
    hyperparams_complex_framework = parse_hyperparams(filepath)

    train_loader_gnn_model_complex_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_complex_framework.get('neighbours_size'),
        batch_size=64,
        input_nodes=train_mask
    )

    val_loader_gnn_model_complex_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_complex_framework.get('neighbours_size'),
        batch_size=64,
        input_nodes=val_mask
    )

    test_loader_gnn_model_complex_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_complex_framework.get('neighbours_size'),
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
        hidden_channels=hyperparams_complex_framework.get('hidden_channels'),
        num_layers=hyperparams_complex_framework.get('num_layers'),
        out_channels=2,
        dropout=hyperparams_complex_framework.get('dropout'),
        act=hyperparams_complex_framework.get('act'),
        aggr=hyperparams_complex_framework.get('aggr'),
        norm=get_norm(hyperparams_complex_framework.get('norm'), hyperparams_complex_framework.get('hidden_channels'))
    )

    gnn_model_complex_framework_without_front_flex = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework_without_front_flex = torch.optim.Adam(
        gnn_model_complex_framework_without_front_flex.parameters(),
        lr=hyperparams_complex_framework.get('lr'), weight_decay=hyperparams_complex_framework.get('weight_decay'))
    criterion_gnn_complex_framework_without_front_flex = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GRAPHSAGE----"""

    filepath = os.path.join(models_finetuning_path,
                            f'graphsage_finetuning_train_set_size_{n_samples_train}.txt')
    hyperparams_graphsage = parse_hyperparams(filepath)

    train_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_graphsage.get('neighbours_size'),
        batch_size=32,
        input_nodes=train_mask
    )

    val_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_graphsage.get('neighbours_size'),
        batch_size=32,
        input_nodes=val_mask
    )

    test_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_graphsage.get('neighbours_size'),
        batch_size=32,
        input_nodes=data.test_mask
    )

    gnn_model_simple_graphsage = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=hyperparams_graphsage.get('hidden_channels'),
        num_layers=hyperparams_graphsage.get('num_layers'),
        out_channels=2,
        dropout=hyperparams_graphsage.get('dropout'),
        act=hyperparams_graphsage.get('act'),
        aggr=hyperparams_graphsage.get('aggr'),
        norm=get_norm(hyperparams_graphsage.get('norm'), hyperparams_graphsage.get('hidden_channels'))
    )
    
    optimizer_gnn_simple_graphsage = torch.optim.Adam(gnn_model_simple_graphsage.parameters(), lr=hyperparams_graphsage.get('lr'),
                                                      weight_decay=hyperparams_graphsage.get('weight_decay'))
    criterion_gnn_simple_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GIN----"""

    filepath = os.path.join(models_finetuning_path,
                            f'gin_finetuning_train_set_size_{n_samples_train}.txt')
    hyperparams_gin = parse_hyperparams(filepath)

    train_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_gin.get('neighbours_size'),
        batch_size=32,
        input_nodes=train_mask
    )

    val_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_gin.get('neighbours_size'),
        batch_size=32,
        input_nodes=val_mask
    )

    test_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=hyperparams_gin.get('neighbours_size'),
        batch_size=32,
        input_nodes=data.test_mask
    )

    gnn_model_simple_gin = GIN(
        in_channels=data.num_features,
        hidden_channels=hyperparams_gin.get('hidden_channels'),
        num_layers=hyperparams_gin.get('num_layers'),
        out_channels=2,
        dropout=hyperparams_gin.get('dropout'),
        act=hyperparams_gin.get('act'),
        norm=get_norm(hyperparams_gin.get('norm'), hyperparams_gin.get('hidden_channels'))
    )

    optimizer_gnn_simple_gin = torch.optim.Adam(gnn_model_simple_gin.parameters(), lr=hyperparams_gin.get('lr')
                                                , weight_decay=hyperparams_gin.get('weight_decay'))
    criterion_gnn_simple_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS GIN----"""

    train_loader_gnn_model_complex_framework_gin_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_gin_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_gin_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10, 25],
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
    gnn_model_downstream_framework_without_flipping_layer = GIN(
        in_channels=data.num_features + 128,
        hidden_channels=64,
        num_layers=3,
        out_channels=2,
        norm=BatchNorm(64),
        dropout=0.30740951784650034,
        act='relu'
    )

    gnn_model_complex_framework_gin_without_front_flex = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                    gnn_model_downstream_framework_without_flipping_layer,
                                                                    False)
    optimizer_gnn_complex_framework_gin_without_front_flex = torch.optim.Adam(
        gnn_model_complex_framework_gin_without_front_flex.parameters(),
        lr=0.004109607017807342, weight_decay=9.859932788621638e-06)
    criterion_gnn_complex_framework_gin_without_front_flex = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GIN and MLP----"""

    train_loader_gnn_model_simple_framework_gin_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_simple_framework_gin_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_simple_framework_gin_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=64,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework_gin = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1],
                                                  hidden_channels=128, output_channels=128, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_simple_framework_gin.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_simple_framework_gin.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    gnn_model_downstream_simple_framework_gin = GIN(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=2,
        out_channels=512,
        norm=BatchNorm(256),
        dropout=0.48112201802846727,
        act='leaky_relu'
    )

    layer_sizes = [128 + 512] + [128] * 3 + [2]
    mlp = build_mlp(layer_sizes, nn.ReLU, 0.335153551771848)

    gnn_model_simple_framework_gin_without_front_flex = DGIAndGNN(dgi_model_simple_framework_gin,
                                                                  gnn_model_downstream_simple_framework_gin, mlp, False)
    optimizer_gnn_simple_framework_gin_without_front_flex = torch.optim.Adam(
        gnn_model_simple_framework_gin_without_front_flex.parameters(),
        lr=0.0008339630658237018, weight_decay=7.330899899115081e-06)
    criterion_gnn_simple_framework_gin_without_front_flex = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        f'simple_framework_{n_samples_train}': {
            'model': gnn_model_simple_framework_without_front_flex,
            'optimizer': optimizer_gnn_simple_framework_without_front_flex,
            'criterion': criterion_gnn_simple_framework_without_front_flex,
            'train_set': train_loader_gnn_model_simple_framework_without_front_flex,
            'val_set': val_loader_gnn_model_simple_framework_without_front_flex,
            'test_set': test_loader_gnn_model_simple_framework_without_front_flex
        },

        f'complex_framework_without_flex_fronts_{n_samples_train}': {
            'model': gnn_model_complex_framework_without_front_flex,
            'optimizer': optimizer_gnn_complex_framework_without_front_flex,
            'criterion': criterion_gnn_complex_framework_without_front_flex,
            'train_set': train_loader_gnn_model_complex_framework_without_front_flex,
            'val_set': val_loader_gnn_model_complex_framework_without_front_flex,
            'test_set': test_loader_gnn_model_complex_framework_without_front_flex
        },

        f'graphsage_{n_samples_train}': {
            'model': gnn_model_simple_graphsage,
            'optimizer': optimizer_gnn_simple_graphsage,
            'criterion': criterion_gnn_simple_graphsage,
            'train_set': train_loader_gnn_simple_graphsage,
            'val_set': val_loader_gnn_simple_graphsage,
            'test_set': test_loader_gnn_simple_graphsage
        },

        f'gin_{n_samples_train}': {
            'model': gnn_model_simple_gin,
            'optimizer': optimizer_gnn_simple_gin,
            'criterion': criterion_gnn_simple_gin,
            'train_set': train_loader_gnn_simple_gin,
            'val_set': val_loader_gnn_simple_gin,
            'test_set': test_loader_gnn_simple_gin
        },

        f'complex_framework_gin_without_flex_fronts_{n_samples_train}': {
            'model': gnn_model_complex_framework_gin_without_front_flex,
            'optimizer': optimizer_gnn_complex_framework_gin_without_front_flex,
            'criterion': criterion_gnn_complex_framework_gin_without_front_flex,
            'train_set': train_loader_gnn_model_complex_framework_gin_without_front_flex,
            'val_set': val_loader_gnn_model_complex_framework_gin_without_front_flex,
            'test_set': test_loader_gnn_model_complex_framework_gin_without_front_flex
        },

        f'simple_framework_gin_without_flex_fronts_{n_samples_train}': {
            'model': gnn_model_simple_framework_gin_without_front_flex,
            'optimizer': optimizer_gnn_simple_framework_gin_without_front_flex,
            'criterion': criterion_gnn_simple_framework_gin_without_front_flex,
            'train_set': train_loader_gnn_model_simple_framework_gin_without_front_flex,
            'val_set': val_loader_gnn_model_simple_framework_gin_without_front_flex,
            'test_set': test_loader_gnn_model_simple_framework_gin_without_front_flex
        }
    }

    return model_dict