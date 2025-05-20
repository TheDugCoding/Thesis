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


#define here the models to test against the framework

def model_list(data):
    """

    :param data: the dataset that is used
    :return: a dict containing all the gnns to test against the framework
    """

    # list of models  to test
    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP----"""

    train_loader_gnn_model_simple_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_simple_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_simple_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64, encoder=EncoderWithoutFlexFrontsGraphsage(64, 64, 2, 3),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_simple_framework.load_state_dict(torch.load(os.path.join(trained_dgi_model_path, 'modeling_graphsage_unsup_trained.pth')))
    # reset first layer, be sure that the hidden channels are the same in DGI
    dgi_model_simple_framework.encoder.dataset_convs[0] = SAGEConv(4, 64)
    # freeze layers 2 and 3
    for param in dgi_model_simple_framework.encoder.conv2.parameters():
        param.requires_grad = False
    for param in dgi_model_simple_framework.encoder.conv3.parameters():
        param.requires_grad = False

        # define the downstream GNN, it is the same as graphsage elliptic++. However the input is different according to the framework architecture
        # gnn_model = GAT(data.num_features + 64, 64, 3,
        #            8).to(device)

    # same model as in garphsage_elliptic
    gnn_model_downstream_simple_framework = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=3,
        out_channels=64,
    )

    # Define MLP layers for classification
    mlp = nn.Sequential(
        nn.Linear(128, 64),  # Adjust hidden_size as needed
        nn.ReLU(),
        nn.Linear(64, 1),  # Output layer for binary classification
        nn.Sigmoid()  # Sigmoid activation for binary classification
    )

    gnn_model_simple_framework_without_front_flex = DGIAndGNN(dgi_model_simple_framework, dgi_model_simple_framework, mlp, False)
    optimizer_gnn_simple_framework_without_front_flex = torch.optim.Adam(
        gnn_model_simple_framework_without_front_flex.parameters(),
        lr=0.005, weight_decay=5e-4)
    criterion_gnn_simple_framework_without_front_flex = torch.nn.CrossEntropyLoss(ignore_index=-1)

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

    """----GRAPHSAGE TOPOLOGICAL INPUT + DATA INPUT----"""

    data_all_features = copy.deepcopy(data)
    data_all_features.x = torch.cat([data.x, data.topological_features], dim=1)

    train_loader_gnn_all_features_graphsage = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[15, 30],
        batch_size=32,
        input_nodes=data_all_features.train_mask
    )

    val_loader_gnn_all_features_graphsage = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[15, 30],
        batch_size=32,
        input_nodes=data_all_features.val_mask
    )

    test_loader_gnn_all_features_graphsage = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[15, 30],
        batch_size=32,
        input_nodes=data_all_features.test_mask
    )

    gnn_model_all_features_graphsage = GraphSAGE(
        in_channels=data_all_features.num_features,
        hidden_channels=128,
        num_layers=3,
        out_channels=2,
        dropout=0.305092490983976,
        aggr='mean',
        act='gelu'
    )
    optimizer_gnn_all_features_graphsage = torch.optim.Adam(gnn_model_all_features_graphsage.parameters(), lr=0.0028090781886872776, weight_decay=7.190743274035764e-06)
    criterion_gnn_all_features_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GRAPHSAGE----"""

    train_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[5, 5, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[5, 5, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[5, 5, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )
    
    gnn_model_simple_graphsage = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=64,
        num_layers=4,
        out_channels=2,
        norm=BatchNorm(64),
        dropout=0.3345954452426979,
        aggr='mean',
        act='gelu'
    )
    optimizer_gnn_simple_graphsage = torch.optim.Adam(gnn_model_simple_graphsage.parameters(), lr=0.003375390880558204, weight_decay=1.6008081531978712e-05)
    criterion_gnn_simple_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GAT----"""

    train_loader_gnn_simple_gat = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[30, 50],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_simple_gat = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[30, 50],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_simple_gat = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[30, 50],
        batch_size=32,
        input_nodes=data.test_mask
    )
    
    gnn_model_simple_gat = GAT(
        in_channels=data.num_features,
        hidden_channels = 128,
        num_layers = 4,
        out_channels = 2,
        heads = 4,
        norm=LayerNorm(128),
        act='gelu',
        dropout=0.3334679509616918,
    )
    optimizer_gnn_simple_gat = torch.optim.Adam(gnn_model_simple_graphsage.parameters(), lr=0.0005020332510524006, weight_decay=3.377923626414446e-05)
    criterion_gnn_simple_gat = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GIN----"""

    train_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )
    
    gnn_model_simple_gin = GIN(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
        norm=BatchNorm(256),
        dropout=0.5370465660662909,
        act='gelu'
    )

    optimizer_gnn_simple_gin = torch.optim.Adam(gnn_model_simple_gin.parameters(), lr=0.002608140007550387
, weight_decay=2.6710002144362644e-06)
    criterion_gnn_simple_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)


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