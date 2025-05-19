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
    DeepGraphInfomaxWithoutFlexFronts, EncoderWithoutFlexFrontsGraphsage, corruption_without_flex_fronts, corruption_without_flex_fronts_random_graph_corruptor
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

    # list of models  to test
    """----BASELINE----"""
    # as in the baseline of the complex experiment from exp 1 n 1

    train_loader_baseline = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_baseline = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_baseline = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_baseline = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64, encoder=EncoderWithoutFlexFrontsGraphsage(4, 64, 64, 3),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_baseline.load_state_dict(
        torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flipping_layer_only_topo_rabo_ethereum.pth')))
    # freeze layers 2 and 3
    for param in dgi_model_baseline.encoder.conv2.parameters():
        param.requires_grad = False
    for param in dgi_model_baseline.encoder.conv3.parameters():
        param.requires_grad = False

        # define the downstream GNN, it is the same as graphsage elliptic++. However the input is different according to the framework architecture
        # gnn_model = GAT(data.num_features + 64, 64, 3,
        #            8).to(device)

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_framework_baseline = GraphSAGE(
        in_channels=data.num_features + 64,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    )

    gnn_model_framework_baseline = DGIPlusGNN(dgi_model_baseline,
                                                                gnn_model_framework_baseline,
                                                                False)
    optimizer_gnn_framework_baseline = torch.optim.Adam(
        gnn_model_framework_baseline.parameters(),
        lr=0.005, weight_decay=5e-4)
    criterion_gnn_framework_baseline = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """---- ABLATION N1: DGI NOT TRAINED LAYERS FROZEN----"""
    # as in the baseline of the complex experiment from exp 1 n 1

    train_loader_dgi_not_trained_layers_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_dgi_not_trained_layers_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_dgi_not_trained_layers_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_dgi_not_trained_layers_frozen = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64, encoder=EncoderWithoutFlexFrontsGraphsage(4, 64, 64, 3),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)

    # FREEZE ALL THE LAYERS AND DO NOT USE THE PRETRAINED DGI

    # freeze layers 2 and 3
    for param in dgi_model_dgi_not_trained_layers_frozen.encoder.conv1.parameters():
        param.requires_grad = False
    for param in dgi_model_dgi_not_trained_layers_frozen.encoder.conv2.parameters():
        param.requires_grad = False
    for param in dgi_model_dgi_not_trained_layers_frozen.encoder.conv3.parameters():
        param.requires_grad = False

        # define the downstream GNN, it is the same as graphsage elliptic++. However the input is different according to the framework architecture
        # gnn_model = GAT(data.num_features + 64, 64, 3,
        #            8).to(device)

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_framework_dgi_not_trained_layers_frozen = GraphSAGE(
        in_channels=data.num_features + 64,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    )

    gnn_model_framework_dgi_not_trained_layers_frozen = DGIPlusGNN(dgi_model_dgi_not_trained_layers_frozen,
                                              gnn_model_framework_dgi_not_trained_layers_frozen,
                                              False)
    optimizer_gnn_framework_dgi_not_trained_layers_frozen = torch.optim.Adam(
        gnn_model_framework_dgi_not_trained_layers_frozen.parameters(),
        lr=0.005, weight_decay=5e-4)
    criterion_gnn_framework_dgi_not_trained_layers_frozen = torch.nn.CrossEntropyLoss(ignore_index=-1)

    
    """----ABLATION N2: DGI NOT TRAINED LAYERS UNFROZEN----"""
    # as in the baseline of the complex experiment from exp 1 n 1

    train_loader_dgi_not_trained_layers_unfrozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_dgi_not_trained_layers_unfrozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_dgi_not_trained_layers_unfrozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_dgi_not_trained_layers_unfrozen = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64, encoder=EncoderWithoutFlexFrontsGraphsage(4, 64, 64, 3),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)

    # UNFREEZE ALL THE LAYERS AND DO NOT USE THE PRETRAINED DGI

        # define the downstream GNN, it is the same as graphsage elliptic++. However the input is different according to the framework architecture
        # gnn_model = GAT(data.num_features + 64, 64, 3,
        #            8).to(device)

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_framework_dgi_not_trained_layers_unfrozen = GraphSAGE(
        in_channels=data.num_features + 64,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    )

    gnn_model_framework_dgi_not_trained_layers_unfrozen = DGIPlusGNN(dgi_model_dgi_not_trained_layers_unfrozen,
                                                                   gnn_model_framework_dgi_not_trained_layers_unfrozen,
                                                                   False)
    optimizer_gnn_framework_dgi_not_trained_layers_unfrozen = torch.optim.Adam(
        gnn_model_framework_dgi_not_trained_layers_unfrozen.parameters(),
        lr=0.005, weight_decay=5e-4)
    criterion_gnn_framework_dgi_not_trained_layers_unfrozen = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----ABLATION N3: DIFFERENT CORRUPTOR----"""
    # as in the different_corruptor of the complex experiment from exp 1 n 1

    train_loader_different_corruptor = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_different_corruptor = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_different_corruptor = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # CHANGED THE CORRUPTOR!

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_different_corruptor = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64, encoder=EncoderWithoutFlexFrontsGraphsage(4, 64, 64, 3),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts_random_graph_corruptor)
    # load the pretrained parameters
    dgi_model_different_corruptor.load_state_dict(
        torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flipping_layer_only_topo_rabo_ethereum.pth')))
    # freeze layers 2 and 3
    for param in dgi_model_different_corruptor.encoder.conv2.parameters():
        param.requires_grad = False
    for param in dgi_model_different_corruptor.encoder.conv3.parameters():
        param.requires_grad = False

        # define the downstream GNN, it is the same as graphsage elliptic++. However the input is different according to the framework architecture
        # gnn_model = GAT(data.num_features + 64, 64, 3,
        #            8).to(device)

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_framework_different_corruptor = GraphSAGE(
        in_channels=data.num_features + 64,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    )

    gnn_model_framework_different_corruptor = DGIPlusGNN(dgi_model_different_corruptor,
                                              gnn_model_framework_different_corruptor,
                                              False)
    optimizer_gnn_framework_different_corruptor = torch.optim.Adam(
        gnn_model_framework_different_corruptor.parameters(),
        lr=0.005, weight_decay=5e-4)
    criterion_gnn_framework_different_corruptor = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----ABLATIONS N4: DIFFERENT POOLING----"""
    # as in the baseline of the complex experiment from exp 1 n 1

    train_loader_different_pooling = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_different_pooling = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_different_pooling = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_different_pooling = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64, encoder=EncoderWithoutFlexFrontsGraphsage(4, 64, 64, 3),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.max(dim=0)[0]),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_different_pooling.load_state_dict(
        torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flipping_layer_only_topo_rabo_ethereum.pth')))
    # freeze layers 2 and 3
    for param in dgi_model_different_pooling.encoder.conv2.parameters():
        param.requires_grad = False
    for param in dgi_model_different_pooling.encoder.conv3.parameters():
        param.requires_grad = False

        # define the downstream GNN, it is the same as graphsage elliptic++. However the input is different according to the framework architecture
        # gnn_model = GAT(data.num_features + 64, 64, 3,
        #            8).to(device)

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_framework_different_pooling = GraphSAGE(
        in_channels=data.num_features + 64,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    )

    gnn_model_framework_different_pooling = DGIPlusGNN(dgi_model_different_pooling,
                                              gnn_model_framework_different_pooling,
                                              False)
    optimizer_gnn_framework_different_pooling = torch.optim.Adam(
        gnn_model_framework_different_pooling.parameters(),
        lr=0.005, weight_decay=5e-4)
    criterion_gnn_framework_different_pooling = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        'framework_baseline': {
            'model': gnn_model_framework_baseline,
            'optimizer': optimizer_gnn_framework_baseline,
            'criterion': criterion_gnn_framework_baseline,
            'train_set': train_loader_baseline,
            'val_set': val_loader_baseline,
            'test_set': test_loader_baseline
        },

        'framework_dgi_not_trained_layers_frozen': {
            'model': gnn_model_framework_dgi_not_trained_layers_frozen,
            'optimizer': optimizer_gnn_framework_dgi_not_trained_layers_frozen,
            'criterion': criterion_gnn_framework_dgi_not_trained_layers_frozen,
            'train_set': train_loader_dgi_not_trained_layers_frozen,
            'val_set': val_loader_dgi_not_trained_layers_frozen,
            'test_set': test_loader_dgi_not_trained_layers_frozen
        },

        'framework_dgi_not_trained_layers_unfrozen': {
            'model': gnn_model_framework_dgi_not_trained_layers_unfrozen,
            'optimizer': optimizer_gnn_framework_dgi_not_trained_layers_unfrozen,
            'criterion': criterion_gnn_framework_dgi_not_trained_layers_unfrozen,
            'train_set': train_loader_dgi_not_trained_layers_unfrozen,
            'val_set': val_loader_dgi_not_trained_layers_unfrozen,
            'test_set': test_loader_dgi_not_trained_layers_unfrozen
        },

        'framework_different_corruptor': {
            'model': gnn_model_framework_different_corruptor,
            'optimizer': optimizer_gnn_framework_different_corruptor,
            'criterion': criterion_gnn_framework_different_corruptor,
            'train_set': train_loader_different_corruptor,
            'val_set': val_loader_different_corruptor,
            'test_set': test_loader_different_corruptor
        }

    }

    return model_dict