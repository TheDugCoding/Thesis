import os

import torch
from torch_geometric.nn import GraphSAGE, GAT, GIN
from torch_geometric.nn import SAGEConv

from src.modeling.final_framework.framework import DGIPlusGNN, corruption
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax import DeepGraphInfomaxFlippingLayer, EncoderFlippingLayer
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax_only_topological_features import DeepGraphInfomaxWithoutFlexFronts, EncoderWithoutFlexFrontsGraphsage
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/testing/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)


#define here the models to test against the framework

def model_list(data):
    """

    :param data: the dataset that is used
    :return: a dict containing all the gnns to test against the framework
    """

    #list of models  to test

    """----FRAMEWORK WITHOUT FLIPPING LAYER----"""
    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64, encoder=EncoderWithoutFlexFrontsGraphsage(4, 64, 64),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flipping_layer_only_topo_rabo_ethereum.pth')))
    # freeze every parameter
    for param in dgi_model_without_flipping_layer.encoder.conv1.parameters():
        param.requires_grad = False
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

    gnn_model_framework_without_flipping_layer = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                         gnn_model_downstream_framework_without_flipping_layer, False)
    optimizer_gnn_framework_without_flipping_layer = torch.optim.Adam(gnn_model_framework_without_flipping_layer.parameters(),
                                                                   lr=0.005, weight_decay=5e-4)
    criterion_gnn_framework_without_flipping_layer = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----FRAMEWORK WITH FLIPPING LAYER----"""
    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_with_flipping_layer = DeepGraphInfomaxFlippingLayer(
        hidden_channels=64, encoder=EncoderFlippingLayer(64, 64, 2),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption)
    # load the pretrained parameters
    dgi_model_with_flipping_layer.load_state_dict(torch.load(os.path.join(trained_dgi_model_path, 'modeling_graphsage_unsup_trained.pth')))
    # reset first layer, be sure that the hidden channels are the same in DGI
    dgi_model_with_flipping_layer.encoder.dataset_convs[0] = SAGEConv(4, 64)
    # freeze layers 2 and 3, let layer 1 learn
    for param in dgi_model_with_flipping_layer.encoder.conv2.parameters():
        param.requires_grad = False
    for param in dgi_model_with_flipping_layer.encoder.conv3.parameters():
        param.requires_grad = False

        # define the downstream GNN, it is the same as graphsage elliptic++. However the input is different according to the framework architecture
        # gnn_model = GAT(data.num_features + 64, 64, 3,
        #            8).to(device)

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_with_flipping_layer = GraphSAGE(
        in_channels=data.num_features + 64,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    )

    gnn_model_framework_with_flipping_layer = DGIPlusGNN(dgi_model_with_flipping_layer, gnn_model_downstream_framework_with_flipping_layer, True)
    optimizer_gnn_framework_with_flipping_layer = torch.optim.Adam(gnn_model_framework_with_flipping_layer.parameters(), lr=0.005, weight_decay=5e-4)
    criterion_gnn_framework_with_flipping_layer = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GRAPHSAGE----"""
    gnn_model_graphsage = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    )
    optimizer_gnn_graphsage = torch.optim.Adam(gnn_model_graphsage.parameters(), lr=0.005, weight_decay=5e-4)
    criterion_gnn_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GAT----"""
    gnn_model_gat = GAT(
        in_channels=data.num_features,
        hidden_channels = 256,
        num_layers = 2,
        out_channels = 2,
        heads = 8)
    optimizer_gnn_gat = torch.optim.Adam(gnn_model_graphsage.parameters(), lr=0.005, weight_decay=5e-4)
    criterion_gnn_gat = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GIN----"""
    gnn_model_gin = GIN(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=2,
        out_channels=2,
    )

    optimizer_gnn_gin = torch.optim.Adam(gnn_model_gin.parameters(), lr=0.005, weight_decay=5e-4)
    criterion_gnn_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)


    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        'framework_without_flipping_layer': {
            'model': gnn_model_framework_without_flipping_layer,
            'optimizer': optimizer_gnn_framework_without_flipping_layer,
            'criterion': criterion_gnn_framework_without_flipping_layer,
        },

        'framework_with_flipping_layer': {
            'model': gnn_model_framework_with_flipping_layer,
            'optimizer': optimizer_gnn_framework_with_flipping_layer,
            'criterion': criterion_gnn_framework_with_flipping_layer,
        },

        'graphsage': {
            'model': gnn_model_graphsage,
            'optimizer': optimizer_gnn_graphsage,
            'criterion': criterion_gnn_graphsage
        },

        'gat': {
            'model': gnn_model_gat,
            'optimizer': optimizer_gnn_gat,
            'criterion': criterion_gnn_gat
        },

        'gin': {
            'model': gnn_model_gin,
            'optimizer': optimizer_gnn_gin,
            'criterion': criterion_gnn_gin
        }
    }


    return model_dict