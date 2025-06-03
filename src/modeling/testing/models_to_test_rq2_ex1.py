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

    # list of models to test
    """----Graphsage and MLP----"""
    train_loader_gnn_model_graphsage_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[15, 30],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_graphsage_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[15, 30],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_graphsage_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[15, 30],
        batch_size=32,
        input_nodes=data.test_mask
    )

    gnn_model_graphsage_and_mlp = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=64,
        num_layers=3,
        out_channels=64,
        norm=LayerNorm(64),
        dropout=0.24535711905702512,
        aggr='max',
        act='gelu'
    )

    # Define MLP layers for classification
    mlp = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )

    gnn_model_graphsage_and_mlp = GraphsageWithMLP(gnn_model_graphsage_and_mlp, mlp)
    optimizer_gnn_model_graphsage_and_mlp = torch.optim.Adam(
        gnn_model_graphsage_and_mlp.parameters(),
        lr=0.002562231015359896, weight_decay=1.0133062540360388e-06)
    criterion_gnn_model_graphsage_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # list of models  to test
    """----DGI and MLP----"""
    train_loader_gnn_model_dgi_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_dgi_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_dgi_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_dgi_and_mlp = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1],
                                                  hidden_channels=128, output_channels=128, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_dgi_and_mlp.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_dgi_and_mlp.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Define MLP layers for classification
    mlp = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )

    gnn_model_model_dgi_and_mlp = DGIWithMLP(dgi_model_dgi_and_mlp, mlp)
    optimizer_gnn_model_dgi_and_mlp = torch.optim.Adam(
        gnn_model_model_dgi_and_mlp.parameters(),
        lr=0.005, weight_decay=5e-4)
    criterion_gnn_model_dgi_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP----"""

    train_loader_gnn_model_simple_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_simple_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_simple_framework_without_front_flex = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
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
        hidden_channels=64,
        num_layers=3,
        out_channels=512,
        dropout=0.39377319491373913,
        act='relu',
        aggr='mean'
    )

    layer_sizes = [128 + 512] + [128] * 3 + [2]
    mlp = build_mlp(layer_sizes, nn.ReLU, 0.39377319491373913)

    gnn_model_simple_framework_without_front_flex = DGIAndGNN(dgi_model_simple_framework,
                                                              gnn_model_downstream_simple_framework, mlp, False)
    optimizer_gnn_simple_framework_without_front_flex = torch.optim.Adam(
        gnn_model_simple_framework_without_front_flex.parameters(),
        lr=0.005, weight_decay=5e-4)
    criterion_gnn_simple_framework_without_front_flex = torch.nn.CrossEntropyLoss(ignore_index=-1)

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
    optimizer_gnn_all_features_graphsage = torch.optim.Adam(gnn_model_all_features_graphsage.parameters(),
                                                            lr=0.0028090781886872776,
                                                            weight_decay=7.190743274035764e-06)
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
    optimizer_gnn_simple_graphsage = torch.optim.Adam(gnn_model_simple_graphsage.parameters(), lr=0.003375390880558204,
                                                      weight_decay=1.6008081531978712e-05)
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
        hidden_channels=128,
        num_layers=4,
        out_channels=2,
        heads=4,
        norm=LayerNorm(128),
        act='gelu',
        dropout=0.3334679509616918,
    )
    optimizer_gnn_simple_gat = torch.optim.Adam(gnn_model_simple_graphsage.parameters(), lr=0.0005020332510524006,
                                                weight_decay=3.377923626414446e-05)
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

    """----GIN TOPOLOGICAL INPUT + DATA INPUT----"""

    data_all_features = copy.deepcopy(data)
    data_all_features.x = torch.cat([data.x, data.topological_features], dim=1)

    train_loader_gnn_all_features_gin = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[10, 20, 30, 40],
        batch_size=32,
        input_nodes=data_all_features.train_mask
    )

    val_loader_gnn_all_features_gin = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[10, 20, 30, 40],
        batch_size=32,
        input_nodes=data_all_features.val_mask
    )

    test_loader_gnn_all_features_gin = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[10, 20, 30, 40],
        batch_size=32,
        input_nodes=data_all_features.test_mask
    )

    gnn_model_all_features_gin = GIN(
        in_channels=data_all_features.num_features,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
        norm=BatchNorm(256),
        dropout=0.29595638014601744,
        act='gelu'
    )

    optimizer_gnn_all_features_gin = torch.optim.Adam(gnn_model_all_features_gin.parameters(), lr=0.0005248031697306322
                                                      , weight_decay=1.576197498517137e-05)
    criterion_gnn_all_features_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        'graphsage_and_mlp': {
            'model': gnn_model_graphsage_and_mlp,
            'optimizer': optimizer_gnn_model_graphsage_and_mlp,
            'criterion': criterion_gnn_model_graphsage_and_mlp,
            'train_set': train_loader_gnn_model_graphsage_and_mlp,
            'val_set': val_loader_gnn_model_graphsage_and_mlp,
            'test_set': test_loader_gnn_model_graphsage_and_mlp,
        },

        'framework_dgi_and_mlp': {
            'model': gnn_model_model_dgi_and_mlp,
            'optimizer': optimizer_gnn_model_dgi_and_mlp,
            'criterion': criterion_gnn_model_dgi_and_mlp,
            'train_set': train_loader_gnn_model_dgi_and_mlp,
            'val_set': val_loader_gnn_model_dgi_and_mlp,
            'test_set': test_loader_gnn_model_dgi_and_mlp
        },

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

        'complex_framework_without_flex_fronts_first_layer_not_frozen': {
            'model': gnn_model_complex_framework_without_front_flex_first_layer_not_frozen,
            'optimizer': optimizer_gnn_complex_framework_without_front_flex_first_layer_not_frozen,
            'criterion': criterion_gnn_complex_framework_without_front_flex_first_layer_not_frozen,
            'train_set': train_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen,
            'val_set': val_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen,
            'test_set': test_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen
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
        },

        'gin_all_features': {
            'model': gnn_model_all_features_gin,
            'optimizer': optimizer_gnn_all_features_gin,
            'criterion': criterion_gnn_all_features_gin,
            'train_set': train_loader_gnn_all_features_gin,
            'val_set': val_loader_gnn_all_features_gin,
            'test_set': test_loader_gnn_all_features_gin
        }
    }

    return model_dict