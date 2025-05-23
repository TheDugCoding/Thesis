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
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/testing/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)


# define here the models to test against the framework

def model_list(data):
    """

    :param data: the dataset that is used
    :return: a dict containing all the gnns to test against the framework
    """


    """----FRAMEWORK WITHOUT FLEX FRONTS----"""

    train_loader_gnn_model_complex_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64, encoder=EncoderWithoutFlexFrontsGraphsage(4, 64, 64, 3),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flipping_layer_only_topo_rabo_ethereum.pth')))
    # freeze layers 2 and 3
    for param in dgi_model_without_flipping_layer.encoder.conv2.parameters():
        param.requires_grad = False
    for param in dgi_model_without_flipping_layer.encoder.conv3.parameters():
        param.requires_grad = False

        # define the downstream GNN, it is the same as graphsage elliptic++. However the input is different according to the framework architecture
        # gnn_model = GAT(data.num_features + 64, 64, 3,
        #            8).to(device)

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer = GraphSAGE(
        in_channels=data.num_features + 64,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    )

    gnn_model_complex_framework_without_front_flex = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework_without_front_flex = torch.optim.Adam(
        gnn_model_complex_framework_without_front_flex.parameters(),
        lr=0.005, weight_decay=5e-4)
    criterion_gnn_complex_framework_without_front_flex = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        'simple_framework_without_flex_fronts': {
            'model': gnn_model_simple_framework_without_front_flex,
            'optimizer': optimizer_gnn_simple_framework_without_front_flex,
            'criterion': criterion_gnn_simple_framework_without_front_flex,
            'train_set': train_loader_gnn_model_simple_framework_without_front_flex,
            'val_set': val_loader_gnn_model_simple_framework_without_front_flex,
            'test_set': test_loader_gnn_model_simple_framework_without_front_flex
        },

        'complex_framework_without_flex_fronts': {
            'model': gnn_model_complex_framework_without_front_flex,
            'optimizer': optimizer_gnn_complex_framework_without_front_flex,
            'criterion': criterion_gnn_complex_framework_without_front_flex,
            'train_set': train_loader_gnn_model_complex_framework_without_front_flex,
            'val_set': val_loader_gnn_model_complex_framework_without_front_flex,
            'test_set': test_loader_gnn_model_complex_framework_without_front_flex
        },

        'graphsage_all_features': {
            'model': gnn_model_all_features_graphsage,
            'optimizer': optimizer_gnn_all_features_graphsage,
            'criterion': criterion_gnn_all_features_graphsage,
            'train_set': train_loader_gnn_all_features_graphsage,
            'val_set': val_loader_gnn_all_features_graphsage,
            'test_set': test_loader_gnn_all_features_graphsage
        },

        'graphsage': {
            'model': gnn_model_simple_graphsage,
            'optimizer': optimizer_gnn_simple_graphsage,
            'criterion': criterion_gnn_simple_graphsage,
            'train_set': train_loader_gnn_simple_graphsage,
            'val_set': val_loader_gnn_simple_graphsage,
            'test_set': test_loader_gnn_simple_graphsage
        },

        'gat': {
            'model': gnn_model_simple_gat,
            'optimizer': optimizer_gnn_simple_gat,
            'criterion': criterion_gnn_simple_gat,
            'train_set': train_loader_gnn_simple_gat,
            'val_set': val_loader_gnn_simple_gat,
            'test_set': test_loader_gnn_simple_gat
        },

        'gin': {
            'model': gnn_model_simple_gin,
            'optimizer': optimizer_gnn_simple_gin,
            'criterion': criterion_gnn_simple_gin,
            'train_set': train_loader_gnn_simple_gin,
            'val_set': val_loader_gnn_simple_gin,
            'test_set': test_loader_gnn_simple_gin
        }
    }

    return model_dict