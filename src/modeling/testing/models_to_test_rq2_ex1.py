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
    DeepGraphInfomaxWithoutFlexFronts, EncoderWithoutFlexFrontsGraphsage, corruption_without_flex_fronts, EncoderWithoutFlexFrontsGIN
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

    train_loader_gnn_model_complex_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[5, 5, 10],
        batch_size=128,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[5, 5, 10],
        batch_size=128,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[5, 5, 10],
        batch_size=128,
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
        torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_without_flipping_layer.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer = GraphSAGE(
        in_channels=data.num_features + 128,
        hidden_channels=128,
        num_layers=3,
        out_channels=2,
        dropout=0.43960265115841607,
        act='relu',
        aggr='mean',
        norm=BatchNorm(128)
    )

    gnn_model_complex_framework = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework = torch.optim.Adam(
        gnn_model_complex_framework.parameters(),
        lr=0.0001781660288878494, weight_decay=0.00048693914641231314)
    criterion_gnn_complex_framework = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS FREE neighbors----"""

    train_loader_gnn_model_complex_framework_free_neighbours = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[5, 5, 10],
        batch_size=128,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_free_neighbours = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[5, 5, 10],
        batch_size=128,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_free_neighbours = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[5, 5, 10],
        batch_size=128,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer_free_neighbours = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1],
                                                  hidden_channels=128, output_channels=128, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer_free_neighbours.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_without_flipping_layer_free_neighbours.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer_free_neighbours = GraphSAGE(
        in_channels=data.num_features + 128,
        hidden_channels=128,
        num_layers=3,
        out_channels=2,
        dropout=0.43960265115841607,
        act='relu',
        aggr='mean',
        norm=BatchNorm(128)
    )

    gnn_model_complex_framework_free_neighbours = DGIPlusGNN(dgi_model_without_flipping_layer_free_neighbours,
                                             gnn_model_downstream_framework_without_flipping_layer_free_neighbours,
                                             False)
    optimizer_gnn_complex_framework_free_neighbours = torch.optim.Adam(
        gnn_model_complex_framework_free_neighbours.parameters(),
        lr=0.0001781660288878494, weight_decay=0.00048693914641231314)
    criterion_gnn_complex_framework_free_neighbours = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS FIRST LAYER UNFREEZE----"""

    train_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=256,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=256,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_without_front_flex_first_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=256,
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
        dropout=0.45137619957974573,
        act='leaky_relu',
        aggr='mean',
        norm=BatchNorm(256)
    )

    gnn_model_complex_framework_without_front_flex_first_layer_not_frozen = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                                       gnn_model_downstream_framework_without_flipping_layer,
                                                                                       False)
    optimizer_gnn_complex_framework_without_front_flex_first_layer_not_frozen = torch.optim.Adam(
        gnn_model_complex_framework_without_front_flex_first_layer_not_frozen.parameters(),
        lr=0.0028937590980406764, weight_decay=5.043122266147557e-06)
    criterion_gnn_complex_framework_without_front_flex_first_layer_not_frozen = torch.nn.CrossEntropyLoss(
        ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS LAST LAYER UNFREEZE----"""

    train_loader_gnn_model_complex_framework_without_front_flex_last_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=256,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_without_front_flex_last_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=256,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_without_front_flex_last_layer_not_frozen = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=256,
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
        num_layers=3,
        out_channels=2,
        dropout=0.43196895784900224,
        act='leaky_relu',
        aggr='mean',
    )

    gnn_model_complex_framework_without_front_flex_last_layer_not_frozen = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                                       gnn_model_downstream_framework_without_flipping_layer,
                                                                                       False)
    optimizer_gnn_complex_framework_without_front_flex_last_layer_not_frozen = torch.optim.Adam(
        gnn_model_complex_framework_without_front_flex_last_layer_not_frozen.parameters(),
        lr=0.001853192775495936, weight_decay=1.9638929703388476e-06)
    criterion_gnn_complex_framework_without_front_flex_last_layer_not_frozen = torch.nn.CrossEntropyLoss(
        ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS GIN ENCODER----"""

    train_loader_gnn_model_complex_framework_without_front_flex_GIN = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_without_front_flex_GIN = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_without_front_flex_GIN = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer_GIN = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=512,
        encoder=EncoderWithoutFlexFrontsGIN(input_channels=data.topological_features.shape[1],
                                                  hidden_channels=128, output_channels=512, layers=4,
                                                  activation_fn=torch.nn.GELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer_GIN.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path, 'modeling_dgi_GIN_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_without_flipping_layer_GIN.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    gnn_model_downstream_framework_without_flipping_layer_GIN = GraphSAGE(
        in_channels=data.num_features + 512,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
        dropout=0.26328036840734287,
        act='leaky_relu',
        aggr='mean',
    )

    gnn_model_complex_framework_without_front_flex_GIN = DGIPlusGNN(dgi_model_without_flipping_layer_GIN,
                                                                gnn_model_downstream_framework_without_flipping_layer_GIN,
                                                                False)
    optimizer_gnn_complex_framework_without_front_flex_GIN = torch.optim.Adam(
        gnn_model_complex_framework_without_front_flex_GIN.parameters(),
        lr=0.00034112978546607924, weight_decay=0.000385338938483396)
    criterion_gnn_complex_framework_without_front_flex_GIN = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS INFONCE----"""

    train_loader_gnn_model_complex_framework_without_front_flex_INFONCE = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=128,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_without_front_flex_INFONCE = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=128,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_without_front_flex_INFONCE = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=128,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer_INFONCE = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1],
                                                  hidden_channels=64, output_channels=64, layers=4,
                                                  activation_fn=torch.nn.ReLU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer_INFONCE.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path, 'modeling_dgi_GraphSage_no_flex_front_only_topo_rabo_ecr_20_infonce.pth')))

    for layer in dgi_model_without_flipping_layer_INFONCE.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer_INFONCE = GraphSAGE(
        in_channels=data.num_features + 64,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
        dropout=0.27540325451229625,
        act='leaky_relu',
        aggr='mean',
    )

    gnn_model_complex_framework_without_front_flex_INFONCE = DGIPlusGNN(dgi_model_without_flipping_layer_INFONCE,
                                                                gnn_model_downstream_framework_without_flipping_layer_INFONCE,
                                                                False)
    optimizer_gnn_complex_framework_without_front_flex_INFONCE = torch.optim.Adam(
        gnn_model_complex_framework_without_front_flex_INFONCE.parameters(),
        lr=0.0019057881038879742, weight_decay=1.4662850459106738e-05)
    criterion_gnn_complex_framework_without_front_flex_INFONCE = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    
    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        'complex_framework_without_flex_fronts': {
            'model': gnn_model_complex_framework,
            'optimizer': optimizer_gnn_complex_framework,
            'criterion': criterion_gnn_complex_framework,
            'train_set': train_loader_gnn_model_complex_framework,
            'val_set': val_loader_gnn_model_complex_framework,
            'test_set': test_loader_gnn_model_complex_framework
        },

        'complex_framework_without_flex_fronts_free_neighbours': {
            'model': gnn_model_complex_framework_free_neighbours,
            'optimizer': optimizer_gnn_complex_framework_free_neighbours,
            'criterion': criterion_gnn_complex_framework_free_neighbours,
            'train_set': train_loader_gnn_model_complex_framework_free_neighbours,
            'val_set': val_loader_gnn_model_complex_framework_free_neighbours,
            'test_set': test_loader_gnn_model_complex_framework_free_neighbours
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
        },

        'complex_framework_without_flex_fronts_GIN_encoder': {
            'model': gnn_model_complex_framework_without_front_flex_GIN,
            'optimizer': optimizer_gnn_complex_framework_without_front_flex_GIN,
            'criterion': criterion_gnn_complex_framework_without_front_flex_GIN,
            'train_set': train_loader_gnn_model_complex_framework_without_front_flex_GIN,
            'val_set': val_loader_gnn_model_complex_framework_without_front_flex_GIN,
            'test_set': test_loader_gnn_model_complex_framework_without_front_flex_GIN
        },

        'complex_framework_without_flex_fronts_INFONCE': {
            'model': gnn_model_complex_framework_without_front_flex_INFONCE,
            'optimizer': optimizer_gnn_complex_framework_without_front_flex_INFONCE,
            'criterion': criterion_gnn_complex_framework_without_front_flex_INFONCE,
            'train_set': train_loader_gnn_model_complex_framework_without_front_flex_INFONCE,
            'val_set': val_loader_gnn_model_complex_framework_without_front_flex_INFONCE,
            'test_set': test_loader_gnn_model_complex_framework_without_front_flex_INFONCE
        }
        
        
        
    }

    return model_dict

#change