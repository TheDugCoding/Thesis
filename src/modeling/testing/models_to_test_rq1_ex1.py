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





    #define here the models to test against the framework

def model_list_rq1_ex1(data):
    """
    :param data: the dataset that is used
    :return: a dict containing all the gnns to test against the framework
    """

    #list of models to test
    """----Graphsage and MLP----"""
    data_gnn_model_graphsage_and_mlp = data
    num_neighbors_gnn_model_graphsage_and_mlp = [5, 5, 10]
    batch_size_gnn_model_graphsage_and_mlp = 32

    gnn_model_graphsage_and_mlp = GraphSAGE(
        in_channels=data[0].num_features,
        hidden_channels=64,
        num_layers=2,
        out_channels=128,
        norm=LayerNorm(64),
        dropout=0.30438159745230026,
        aggr='max',
        act='gelu',
    )

    # Define MLP layers for classification
    mlp = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )

    gnn_model_graphsage_and_mlp = GraphsageWithMLP(gnn_model_graphsage_and_mlp, mlp)
    optimizer_gnn_model_graphsage_and_mlp = torch.optim.Adam(
        gnn_model_graphsage_and_mlp.parameters(),
        lr=0.0008820290413909019, weight_decay=5.469491667151833e-06)
    criterion_gnn_model_graphsage_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # list of models  to test
    """----DGI and MLP----"""
    data_gnn_model_dgi_and_mlp = data
    num_neighbors_gnn_model_dgi_and_mlp = [10, 20, 40]
    batch_size_gnn_model_dgi_and_mlp = 32

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_dgi_and_mlp = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128, encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1], hidden_channels=128, output_channels=128, layers=4, activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_dgi_and_mlp.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_dgi_and_mlp.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Define MLP layers for classification
    layer_sizes = [128] + [128] * 3 + [2]
    mlp_dgi_and_mlp = build_mlp(layer_sizes, nn.LeakyReLU, 0.2933583126562131)

    gnn_model_model_dgi_and_mlp = DGIWithMLP(dgi_model_dgi_and_mlp, mlp_dgi_and_mlp)
    optimizer_gnn_model_dgi_and_mlp = torch.optim.Adam(
        gnn_model_model_dgi_and_mlp.parameters(),
        lr=0.00022972467644955108, weight_decay=1.9482676332552186e-05)
    criterion_gnn_model_dgi_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP----"""
    data_gnn_model_simple_framework = data
    num_neighbors_gnn_model_simple_framework = [30, 50]
    batch_size_gnn_model_simple_framework = 256

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1],
                                                  hidden_channels=128, output_channels=128, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_simple_framework.load_state_dict(torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_simple_framework.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in garphsage_elliptic
    gnn_model_downstream_simple_framework = GraphSAGE(
        in_channels=data[0].num_features,
        hidden_channels=128,
        num_layers=3,
        out_channels=256,
        dropout=0.4101899872847951,
        act='relu',
        aggr='mean',
        norm=BatchNorm(128)
    )

    layer_sizes = [128 + 256] + [128] * 2 + [2]
    mlp_simple_framework = build_mlp(layer_sizes, nn.ELU, 0.47220891507456414)

    gnn_model_simple_framework = DGIAndGNN(dgi_model_simple_framework, gnn_model_downstream_simple_framework, mlp_simple_framework, False)
    optimizer_gnn_simple_framework = torch.optim.Adam(
        gnn_model_simple_framework.parameters(),
        lr=0.003661856103606063, weight_decay=6.0767598595010515e-06)
    criterion_gnn_simple_framework = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP ONLY DEGREE----"""
    data_gnn_model_simple_framework_only_degree = []
    for d in data:
        d_copy = copy.deepcopy(d)
        d_copy.topological_features = d_copy.topological_features[:, 0].unsqueeze(-1)
        data_gnn_model_simple_framework_only_degree.append(d_copy)
    num_neighbors_gnn_model_simple_framework_only_degree = [10, 10, 25]
    batch_size_gnn_model_simple_framework_only_degree = 256

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework_only_degree = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=32,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_gnn_model_simple_framework_only_degree[0].topological_features.shape[1],
                                                  hidden_channels=64, output_channels=32, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_simple_framework_only_degree.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_GraphSage_no_flex_front_only_topo_rabo_ecr_20_only_degree.pth')))

    for layer in dgi_model_simple_framework_only_degree.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in garphsage_elliptic
    gnn_model_downstream_simple_framework_only_degree = GraphSAGE(
        in_channels=data[0].num_features,
        hidden_channels=256,
        num_layers=3,
        out_channels=256,
        dropout=0.27054373214864214,
        act='gelu',
        aggr='mean',
        norm=BatchNorm(256)
    )

    layer_sizes = [32 + 256] + [256] * 4 + [2]
    mlp_only_degree = build_mlp(layer_sizes, nn.ReLU, 0.5315785985902932)

    gnn_model_simple_framework_only_degree = DGIAndGNN(dgi_model_simple_framework_only_degree,
                                                              gnn_model_downstream_simple_framework_only_degree, mlp_only_degree, False)
    optimizer_gnn_simple_framework_only_degree = torch.optim.Adam(
        gnn_model_simple_framework_only_degree.parameters(),
        lr=0.00033281329662653713, weight_decay=3.258855114145082e-05)
    criterion_gnn_simple_framework_only_degree = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GIN and MLP----"""
    data_gnn_model_simple_framework_gin = data
    num_neighbors_gnn_model_simple_framework_gin = [10, 20, 40]
    batch_size_gnn_model_simple_framework_gin = 64

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework_gin = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1],
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
        in_channels=data[0].num_features,
        hidden_channels=128,
        num_layers=3,
        out_channels=128,
        norm=LayerNorm(128),
        dropout=0.46085345459997396,
        act='relu'
    )

    layer_sizes = [128 + 128] + [64] * 4 + [2]
    mlp = build_mlp(layer_sizes, nn.GELU, 0.2035389379103658)

    gnn_model_simple_framework_gin = DGIAndGNN(dgi_model_simple_framework_gin,
                                                              gnn_model_downstream_simple_framework_gin, mlp, False)
    optimizer_gnn_simple_framework_gin = torch.optim.Adam(
        gnn_model_simple_framework_gin.parameters(),
        lr=0.00047676396765200795, weight_decay=1.019513255636531e-06)
    criterion_gnn_simple_framework_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS----"""
    data_gnn_model_complex_framework = data
    num_neighbors_gnn_model_complex_framework = [5, 5, 10]
    batch_size_gnn_model_complex_framework = 128

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1],
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
        in_channels=data[0].num_features + 128,
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

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS ONLY DEGREE----"""
    data_gnn_model_complex_framework_only_degree = []
    for d in data:
        d_copy = copy.deepcopy(d)
        d_copy.topological_features = d_copy.topological_features[:, 0].unsqueeze(-1)
        data_gnn_model_complex_framework_only_degree.append(d_copy)
    num_neighbors_gnn_model_complex_framework_only_degree = [10, 10, 25]
    batch_size_gnn_model_complex_framework_only_degree = 128

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer_only_degree = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=32,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_gnn_model_complex_framework_only_degree[0].topological_features.shape[1],
                                                  hidden_channels=64, output_channels=32, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer_only_degree.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path, 'modeling_dgi_GraphSage_no_flex_front_only_topo_rabo_ecr_20_only_degree.pth')))

    for layer in dgi_model_without_flipping_layer_only_degree.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer_only_degree = GraphSAGE(
        in_channels=data[0].num_features + 32,
        hidden_channels=128,
        num_layers=3,
        out_channels=2,
        dropout=0.38374741544679797,
        act='leaky_relu',
        aggr='sum',
    )

    gnn_model_complex_framework_only_degree = DGIPlusGNN(dgi_model_without_flipping_layer_only_degree,
                                                                gnn_model_downstream_framework_without_flipping_layer_only_degree,
                                                                False)
    optimizer_gnn_complex_framework_only_degree = torch.optim.Adam(
        gnn_model_complex_framework_only_degree.parameters(),
        lr=0.0006350752174843652, weight_decay=3.2814488668540757e-06)
    criterion_gnn_complex_framework_only_degree = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS GIN----"""
    data_gnn_model_complex_framework_gin = data
    num_neighbors_gnn_model_complex_framework_gin = [10, 20, 40]
    batch_size_gnn_model_complex_framework_gin = 64

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1],
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
    gnn_model_downstream_framework_without_flipping_layer =  GIN(
        in_channels=data[0].num_features+128,
        hidden_channels=64,
        num_layers=3,
        out_channels=2,
        norm=GraphNorm(64),
        dropout=0.28712777685883895,
        act='relu'
    )

    gnn_model_complex_framework_gin = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework_gin = torch.optim.Adam(
        gnn_model_complex_framework_gin.parameters(),
        lr=0.0020712222975631865, weight_decay=8.60445035335777e-06)
    criterion_gnn_complex_framework_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GRAPHSAGE TOPOLOGICAL INPUT + DATA INPUT----"""
    data_gnn_all_features_graphsage = []
    for d in data:
        d_copy = copy.deepcopy(d)
        d_copy.x = torch.cat([d_copy.x, d_copy.topological_features], dim=1)
        data_gnn_all_features_graphsage.append(d_copy)
    num_neighbors_gnn_all_features_graphsage = [10, 10, 25]
    batch_size_gnn_all_features_graphsage = 32

    gnn_model_all_features_graphsage = GraphSAGE(
        in_channels=data_gnn_all_features_graphsage[0].num_features,
        hidden_channels=64,
        num_layers=3,
        out_channels=2,
        dropout=0.28543014475626854,
        aggr='sum',
        act='relu'
    )
    optimizer_gnn_all_features_graphsage = torch.optim.Adam(gnn_model_all_features_graphsage.parameters(), lr=0.0007538050390367987, weight_decay=0.00011134434716490531)
    criterion_gnn_all_features_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GRAPHSAGE----"""
    data_gnn_simple_graphsage = data
    num_neighbors_gnn_simple_graphsage = [30, 50]
    batch_size_gnn_simple_graphsage = 32

    gnn_model_simple_graphsage = GraphSAGE(
        in_channels=data[0].num_features,
        hidden_channels=256,
        num_layers=4,
        out_channels=2,
        norm=LayerNorm(256),
        dropout=0.25734662570892874,
        aggr='mean',
        act='gelu',
    )
    optimizer_gnn_simple_graphsage = torch.optim.Adam(gnn_model_simple_graphsage.parameters(), lr=0.0005714494050755839, weight_decay=3.2461817790221313e-06)
    criterion_gnn_simple_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GIN----"""
    data_gnn_simple_gin = data
    num_neighbors_gnn_simple_gin = [5, 5, 10]
    batch_size_gnn_simple_gin = 32

    gnn_model_simple_gin = GIN(
        in_channels=data[0].num_features,
        hidden_channels=128,
        num_layers=2,
        out_channels=2,
        dropout=0.31452147882039877,
        act='relu'
    )

    optimizer_gnn_simple_gin = torch.optim.Adam(gnn_model_simple_gin.parameters(), lr=0.0007333023993989535
                                                , weight_decay=4.0777574201816404e-05)
    criterion_gnn_simple_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GIN TOPOLOGICAL INPUT + DATA INPUT----"""
    data_gnn_all_features_gin = []
    for d in data:
        d_copy = copy.deepcopy(d)
        d_copy.x = torch.cat([d_copy.x, d_copy.topological_features], dim=1)
        data_gnn_all_features_gin.append(d_copy)
    num_neighbors_gnn_all_features_gin = [5, 5, 10]
    batch_size_gnn_all_features_gin = 32

    gnn_model_all_features_gin = GIN(
        in_channels=data_gnn_all_features_gin[0].num_features,
        hidden_channels=128,
        num_layers=2,
        out_channels=2,
        norm=LayerNorm(128),
        dropout=0.5727836990311036,
        act='gelu'
    )

    optimizer_gnn_all_features_gin = torch.optim.Adam(gnn_model_all_features_gin.parameters(), lr=0.0049756039260917964
    , weight_decay=1.7765113710402859e-06)
    criterion_gnn_all_features_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)


    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        'GraphSAGE + MLP': {
            'model': gnn_model_graphsage_and_mlp,
            'optimizer': optimizer_gnn_model_graphsage_and_mlp,
            'criterion': criterion_gnn_model_graphsage_and_mlp,
            'data': data_gnn_model_graphsage_and_mlp,
            'num_neighbours': num_neighbors_gnn_model_graphsage_and_mlp,
            'batch_size': batch_size_gnn_model_graphsage_and_mlp,
        },

        'framework_dgi_and_mlp': {
            'model': gnn_model_model_dgi_and_mlp,
            'optimizer': optimizer_gnn_model_dgi_and_mlp,
            'criterion': criterion_gnn_model_dgi_and_mlp,
            'data': data_gnn_model_dgi_and_mlp,
            'num_neighbours': num_neighbors_gnn_model_dgi_and_mlp,
            'batch_size': batch_size_gnn_model_dgi_and_mlp,
        },

        'simple_framework_without_flex_fronts': {
            'model': gnn_model_simple_framework,
            'optimizer': optimizer_gnn_simple_framework,
            'criterion': criterion_gnn_simple_framework,
            'data': data_gnn_model_simple_framework,
            'num_neighbours': num_neighbors_gnn_model_simple_framework,
            'batch_size': batch_size_gnn_model_simple_framework,
        },

        'simple_framework_without_flex_fronts_only_degree': {
            'model': gnn_model_simple_framework_only_degree,
            'optimizer': optimizer_gnn_simple_framework_only_degree,
            'criterion': criterion_gnn_simple_framework_only_degree,
            'data': data_gnn_model_simple_framework_only_degree,
            'num_neighbours': num_neighbors_gnn_model_simple_framework_only_degree,
            'batch_size': batch_size_gnn_model_simple_framework_only_degree,
        },

        'simple_framework_gin_without_flex_fronts': {
            'model': gnn_model_simple_framework_gin,
            'optimizer': optimizer_gnn_simple_framework_gin,
            'criterion': criterion_gnn_simple_framework_gin,
            'data': data_gnn_model_simple_framework_gin,
            'num_neighbours': num_neighbors_gnn_model_simple_framework_gin,
            'batch_size': batch_size_gnn_model_simple_framework_gin,
        },

        'complex_framework_without_flex_fronts': {
            'model': gnn_model_complex_framework,
            'optimizer': optimizer_gnn_complex_framework,
            'criterion': criterion_gnn_complex_framework,
            'data': data_gnn_model_complex_framework,
            'num_neighbours': num_neighbors_gnn_model_complex_framework,
            'batch_size': batch_size_gnn_model_complex_framework,
        },

        'complex_framework_without_flex_fronts_only_degree': {
            'model': gnn_model_complex_framework_only_degree,
            'optimizer': optimizer_gnn_complex_framework_only_degree,
            'criterion': criterion_gnn_complex_framework_only_degree,
            'data': data_gnn_model_complex_framework_only_degree,
            'num_neighbours': num_neighbors_gnn_model_complex_framework_only_degree,
            'batch_size': batch_size_gnn_model_complex_framework_only_degree,
        },

        'complex_framework_gin_without_flex_fronts': {
            'model': gnn_model_complex_framework_gin,
            'optimizer': optimizer_gnn_complex_framework_gin,
            'criterion': criterion_gnn_complex_framework_gin,
            'data': data_gnn_model_complex_framework_gin,
            'num_neighbours': num_neighbors_gnn_model_complex_framework_gin,
            'batch_size': batch_size_gnn_model_complex_framework_gin,
        },

        'graphsage_all_features': {
            'model': gnn_model_all_features_graphsage,
            'optimizer': optimizer_gnn_all_features_graphsage,
            'criterion': criterion_gnn_all_features_graphsage,
            'data': data_gnn_all_features_graphsage,
            'num_neighbours': num_neighbors_gnn_all_features_graphsage,
            'batch_size': batch_size_gnn_all_features_graphsage,
        },

        'graphsage': {
            'model': gnn_model_simple_graphsage,
            'optimizer': optimizer_gnn_simple_graphsage,
            'criterion': criterion_gnn_simple_graphsage,
            'data': data_gnn_simple_graphsage,
            'num_neighbours': num_neighbors_gnn_simple_graphsage,
            'batch_size': batch_size_gnn_simple_graphsage,
        },

        'gin': {
            'model': gnn_model_simple_gin,
            'optimizer': optimizer_gnn_simple_gin,
            'criterion': criterion_gnn_simple_gin,
            'data': data_gnn_simple_gin,
            'num_neighbours': num_neighbors_gnn_simple_gin,
            'batch_size': batch_size_gnn_simple_gin,
        },

        'gin_all_features': {
            'model': gnn_model_all_features_gin,
            'optimizer': optimizer_gnn_all_features_gin,
            'criterion': criterion_gnn_all_features_gin,
            'data': data_gnn_all_features_gin,
            'num_neighbours': num_neighbors_gnn_all_features_gin,
            'batch_size': batch_size_gnn_all_features_gin,
        },
    }


    return model_dict
