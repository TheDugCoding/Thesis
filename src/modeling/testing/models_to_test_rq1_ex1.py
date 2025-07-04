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
    train_loader_gnn_model_graphsage_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_graphsage_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_graphsage_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    gnn_model_graphsage_and_mlp = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=128,
        num_layers=3,
        out_channels=256,
        norm=BatchNorm(128),
        dropout= 0.24878192535912913,
        aggr='mean',
        act='leaky_relu',
    )

    # Define MLP layers for classification
    mlp = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2),
    )

    gnn_model_graphsage_and_mlp = GraphsageWithMLP(gnn_model_graphsage_and_mlp, mlp)
    optimizer_gnn_model_graphsage_and_mlp = torch.optim.Adam(
        gnn_model_graphsage_and_mlp.parameters(),
        lr= 0.0008538676059177102, weight_decay=1.2939685890875305e-05)
    criterion_gnn_model_graphsage_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # list of models  to test
    """----DGI and MLP----"""
    train_loader_gnn_model_dgi_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_dgi_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_dgi_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_dgi_and_mlp = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128, encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1], hidden_channels=128, output_channels=128, layers=4, activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_dgi_and_mlp.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_dgi_and_mlp.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Define MLP layers for classification
    layer_sizes = [128] + [64] * 2 + [2]
    mlp_dgi_and_mlp = build_mlp(layer_sizes, nn.ReLU, 0.2209378396144853)

    gnn_model_model_dgi_and_mlp = DGIWithMLP(dgi_model_dgi_and_mlp, mlp_dgi_and_mlp)
    optimizer_gnn_model_dgi_and_mlp = torch.optim.Adam(
        gnn_model_model_dgi_and_mlp.parameters(),
        lr=0.000194519851846262, weight_decay=8.360189778797015e-06)
    criterion_gnn_model_dgi_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP----"""

    train_loader_gnn_model_simple_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_simple_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_simple_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
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
    dgi_model_simple_framework.load_state_dict(torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_simple_framework.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in garphsage_elliptic
    gnn_model_downstream_simple_framework = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=3,
        out_channels=512,
        dropout=0.5001755031050078,
        act='leaky_relu',
        aggr='max',
        norm=LayerNorm(256)
    )

    layer_sizes = [128 + 512] + [64] * 3 + [2]
    mlp_simple_framework = build_mlp(layer_sizes, nn.LeakyReLU, 0.32199294601431283)

    gnn_model_simple_framework = DGIAndGNN(dgi_model_simple_framework, gnn_model_downstream_simple_framework, mlp_simple_framework, False)
    optimizer_gnn_simple_framework = torch.optim.Adam(
        gnn_model_simple_framework.parameters(),
        lr=0.0007294031543116303, weight_decay=1.1801673829692855e-05)
    criterion_gnn_simple_framework = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP ONLY DEGREE----"""

    data_only_degree = copy.deepcopy(data)
    data_only_degree.topological_features = data_only_degree.topological_features[:, 0].unsqueeze(-1)

    train_loader_gnn_model_simple_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.train_mask
    )

    val_loader_gnn_model_simple_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.val_mask
    )

    test_loader_gnn_model_simple_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework_only_degree = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=32,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_only_degree.topological_features.shape[1],
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
        in_channels=data.num_features,
        hidden_channels=64,
        num_layers=3,
        out_channels=512,
        dropout=0.2025004855748988,
        act='leaky_relu',
        aggr='max',
        norm=BatchNorm(64)
    )

    layer_sizes = [32 + 512] + [256] * 3 + [2]
    mlp_only_degree = build_mlp(layer_sizes, nn.ELU, 0.3307471174677133)

    gnn_model_simple_framework_only_degree = DGIAndGNN(dgi_model_simple_framework_only_degree,
                                                              gnn_model_downstream_simple_framework_only_degree, mlp_only_degree, False)
    optimizer_gnn_simple_framework_only_degree = torch.optim.Adam(
        gnn_model_simple_framework_only_degree.parameters(),
        lr=0.002322321872125943, weight_decay=4.234449204450303e-06)
    criterion_gnn_simple_framework_only_degree = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GIN and MLP----"""

    train_loader_gnn_model_simple_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_simple_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_simple_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
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
        num_layers=3,
        out_channels=512,
        norm=GraphNorm(256),
        dropout=0.2950022096499954,
        act='relu'
    )

    layer_sizes = [128 + 512] + [64] * 4 + [2]
    mlp = build_mlp(layer_sizes, nn.ELU, 0.3163058988828747)

    gnn_model_simple_framework_gin = DGIAndGNN(dgi_model_simple_framework_gin,
                                                              gnn_model_downstream_simple_framework_gin, mlp, False)
    optimizer_gnn_simple_framework_gin = torch.optim.Adam(
        gnn_model_simple_framework_gin.parameters(),
        lr=0.0004445808190363408, weight_decay=0.00010875565406402025)
    criterion_gnn_simple_framework_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS----"""

    train_loader_gnn_model_complex_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
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
        dropout=0.3935595942964136,
        act='gelu',
        aggr='max'
    )

    gnn_model_complex_framework = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework = torch.optim.Adam(
        gnn_model_complex_framework.parameters(),
        lr=0.001576586258891951, weight_decay=1.2813913397210403e-06)
    criterion_gnn_complex_framework = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS ONLY DEGREE----"""

    data_only_degree = copy.deepcopy(data)
    data_only_degree.topological_features = data_only_degree.topological_features[:, 0].unsqueeze(-1)

    train_loader_gnn_model_complex_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.train_mask
    )

    val_loader_gnn_model_complex_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.val_mask
    )

    test_loader_gnn_model_complex_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer_only_degree = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=32,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_only_degree.topological_features.shape[1],
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
        in_channels=data.num_features + 32,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
        dropout=0.21380223621527408,
        act='relu',
        aggr='mean',
        norm=GraphNorm(256)
    )

    gnn_model_complex_framework_only_degree = DGIPlusGNN(dgi_model_without_flipping_layer_only_degree,
                                                                gnn_model_downstream_framework_without_flipping_layer_only_degree,
                                                                False)
    optimizer_gnn_complex_framework_only_degree = torch.optim.Adam(
        gnn_model_complex_framework_only_degree.parameters(),
        lr=0.003249483526301171, weight_decay=1.0420379869125718e-06)
    criterion_gnn_complex_framework_only_degree = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS GIN----"""

    train_loader_gnn_model_complex_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
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
    gnn_model_downstream_framework_without_flipping_layer =  GIN(
        in_channels=data.num_features+128,
        hidden_channels=64,
        num_layers=3,
        out_channels=2,
        norm=GraphNorm(64),
        dropout=0.4685511903169562,
        act='leaky_relu'
    )

    gnn_model_complex_framework_gin = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework_gin = torch.optim.Adam(
        gnn_model_complex_framework_gin.parameters(),
        lr=0.0006401612564081927, weight_decay=3.869991153635051e-05)
    criterion_gnn_complex_framework_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GRAPHSAGE TOPOLOGICAL INPUT + DATA INPUT----"""

    data_all_features = copy.deepcopy(data)
    data_all_features.x = torch.cat([data.x, data.topological_features], dim=1)

    train_loader_gnn_all_features_graphsage = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=32,
        input_nodes=data_all_features.train_mask
    )

    val_loader_gnn_all_features_graphsage = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=32,
        input_nodes=data_all_features.val_mask
    )

    test_loader_gnn_all_features_graphsage = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=32,
        input_nodes=data_all_features.test_mask
    )

    gnn_model_all_features_graphsage = GraphSAGE(
        in_channels=data_all_features.num_features,
        hidden_channels=128,
        num_layers=4,
        out_channels=2,
        dropout=0.2639459463249423,
        aggr='mean',
        act='elu'
    )
    optimizer_gnn_all_features_graphsage = torch.optim.Adam(gnn_model_all_features_graphsage.parameters(), lr=0.00085746256660606, weight_decay=1.995683035575948e-05)
    criterion_gnn_all_features_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GRAPHSAGE----"""

    train_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    gnn_model_simple_graphsage = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=4,
        out_channels=2,
        norm=BatchNorm(256),
        dropout=0.28082173362847057,
        aggr='max',
        act='leaky_relu',
    )
    optimizer_gnn_simple_graphsage = torch.optim.Adam(gnn_model_simple_graphsage.parameters(), lr=0.0018573604269558593, weight_decay=3.498293657545283e-06)
    criterion_gnn_simple_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GIN----"""

    train_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 30, 40],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 30, 40],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 30, 40],
        batch_size=32,
        input_nodes=data.test_mask
    )

    gnn_model_simple_gin = GIN(
        in_channels=data.num_features,
        hidden_channels=64,
        num_layers=2,
        out_channels=2,
        norm=BatchNorm(64),
        dropout=0.3986306469433091,
        act='relu'
    )

    optimizer_gnn_simple_gin = torch.optim.Adam(gnn_model_simple_gin.parameters(), lr=0.0013260343088210972
                                                , weight_decay=1.495957475057153e-05)
    criterion_gnn_simple_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GIN TOPOLOGICAL INPUT + DATA INPUT----"""

    data_all_features = copy.deepcopy(data)
    data_all_features.x = torch.cat([data.x, data.topological_features], dim=1)

    train_loader_gnn_all_features_gin = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[20, 20],
        batch_size=32,
        input_nodes=data_all_features.train_mask
    )

    val_loader_gnn_all_features_gin= NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[20, 20],
        batch_size=32,
        input_nodes=data_all_features.val_mask
    )

    test_loader_gnn_all_features_gin = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[20, 20],
        batch_size=32,
        input_nodes=data_all_features.test_mask
    )

    gnn_model_all_features_gin = GIN(
        in_channels=data_all_features.num_features,
        hidden_channels=128,
        num_layers=3,
        out_channels=2,
        dropout=0.34784199594158705,
        act='leaky_relu'
    )

    optimizer_gnn_all_features_gin = torch.optim.Adam(gnn_model_all_features_gin.parameters(), lr=0.0010138466771643904
    , weight_decay=4.636489953583516e-06)
    criterion_gnn_all_features_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)


    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        'GraphSAGE + MLP': {
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
            'model': gnn_model_simple_framework,
            'optimizer': optimizer_gnn_simple_framework,
            'criterion': criterion_gnn_simple_framework,
            'train_set': train_loader_gnn_model_simple_framework,
            'val_set': val_loader_gnn_model_simple_framework,
            'test_set': test_loader_gnn_model_simple_framework
        },

        'simple_framework_without_flex_fronts_only_degree': {
            'model': gnn_model_simple_framework_only_degree,
            'optimizer': optimizer_gnn_simple_framework_only_degree,
            'criterion': criterion_gnn_simple_framework_only_degree,
            'train_set': train_loader_gnn_model_simple_framework_only_degree,
            'val_set': val_loader_gnn_model_simple_framework_only_degree,
            'test_set': test_loader_gnn_model_simple_framework_only_degree
        },

        'simple_framework_gin_without_flex_fronts': {
            'model': gnn_model_simple_framework_gin,
            'optimizer': optimizer_gnn_simple_framework_gin,
            'criterion': criterion_gnn_simple_framework_gin,
            'train_set': train_loader_gnn_model_simple_framework_gin,
            'val_set': val_loader_gnn_model_simple_framework_gin,
            'test_set': test_loader_gnn_model_simple_framework_gin
        },

        'complex_framework_without_flex_fronts': {
            'model': gnn_model_complex_framework,
            'optimizer': optimizer_gnn_complex_framework,
            'criterion': criterion_gnn_complex_framework,
            'train_set': train_loader_gnn_model_complex_framework,
            'val_set': val_loader_gnn_model_complex_framework,
            'test_set': test_loader_gnn_model_complex_framework
        },

        'complex_framework_without_flex_fronts_only_degree': {
            'model': gnn_model_complex_framework_only_degree,
            'optimizer': optimizer_gnn_complex_framework_only_degree,
            'criterion': criterion_gnn_complex_framework_only_degree,
            'train_set': train_loader_gnn_model_complex_framework_only_degree,
            'val_set': val_loader_gnn_model_complex_framework_only_degree,
            'test_set': test_loader_gnn_model_complex_framework_only_degree
        },

        'complex_framework_gin_without_flex_fronts': {
            'model': gnn_model_complex_framework_gin,
            'optimizer': optimizer_gnn_complex_framework_gin,
            'criterion': criterion_gnn_complex_framework_gin,
            'train_set': train_loader_gnn_model_complex_framework_gin,
            'val_set': val_loader_gnn_model_complex_framework_gin,
            'test_set': test_loader_gnn_model_complex_framework_gin
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

#changeimport copy
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
    train_loader_gnn_model_graphsage_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_graphsage_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_graphsage_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    gnn_model_graphsage_and_mlp = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=128,
        num_layers=3,
        out_channels=256,
        norm=BatchNorm(128),
        dropout= 0.24878192535912913,
        aggr='mean',
        act='leaky_relu',
    )

    # Define MLP layers for classification
    mlp = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2),
    )

    gnn_model_graphsage_and_mlp = GraphsageWithMLP(gnn_model_graphsage_and_mlp, mlp)
    optimizer_gnn_model_graphsage_and_mlp = torch.optim.Adam(
        gnn_model_graphsage_and_mlp.parameters(),
        lr= 0.0008538676059177102, weight_decay=1.2939685890875305e-05)
    criterion_gnn_model_graphsage_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # list of models  to test
    """----DGI and MLP----"""
    train_loader_gnn_model_dgi_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_dgi_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_dgi_and_mlp = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_dgi_and_mlp = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=128, encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data.topological_features.shape[1], hidden_channels=128, output_channels=128, layers=4, activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_dgi_and_mlp.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_dgi_and_mlp.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Define MLP layers for classification
    layer_sizes = [128] + [64] * 2 + [2]
    mlp_dgi_and_mlp = build_mlp(layer_sizes, nn.ReLU, 0.2209378396144853)

    gnn_model_model_dgi_and_mlp = DGIWithMLP(dgi_model_dgi_and_mlp, mlp_dgi_and_mlp)
    optimizer_gnn_model_dgi_and_mlp = torch.optim.Adam(
        gnn_model_model_dgi_and_mlp.parameters(),
        lr=0.000194519851846262, weight_decay=8.360189778797015e-06)
    criterion_gnn_model_dgi_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP----"""

    train_loader_gnn_model_simple_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_simple_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_simple_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
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
    dgi_model_simple_framework.load_state_dict(torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_simple_framework.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # same model as in garphsage_elliptic
    gnn_model_downstream_simple_framework = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=3,
        out_channels=512,
        dropout=0.5001755031050078,
        act='leaky_relu',
        aggr='max',
        norm=LayerNorm(256)
    )

    layer_sizes = [128 + 512] + [64] * 3 + [2]
    mlp_simple_framework = build_mlp(layer_sizes, nn.LeakyReLU, 0.32199294601431283)

    gnn_model_simple_framework = DGIAndGNN(dgi_model_simple_framework, gnn_model_downstream_simple_framework, mlp_simple_framework, False)
    optimizer_gnn_simple_framework = torch.optim.Adam(
        gnn_model_simple_framework.parameters(),
        lr=0.0007294031543116303, weight_decay=1.1801673829692855e-05)
    criterion_gnn_simple_framework = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP ONLY DEGREE----"""

    data_only_degree = copy.deepcopy(data)
    data_only_degree.topological_features = data_only_degree.topological_features[:, 0].unsqueeze(-1)

    train_loader_gnn_model_simple_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.train_mask
    )

    val_loader_gnn_model_simple_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.val_mask
    )

    test_loader_gnn_model_simple_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework_only_degree = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=32,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_only_degree.topological_features.shape[1],
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
        in_channels=data.num_features,
        hidden_channels=64,
        num_layers=3,
        out_channels=512,
        dropout=0.2025004855748988,
        act='leaky_relu',
        aggr='max',
        norm=BatchNorm(64)
    )

    layer_sizes = [32 + 512] + [256] * 3 + [2]
    mlp_only_degree = build_mlp(layer_sizes, nn.ELU, 0.3307471174677133)

    gnn_model_simple_framework_only_degree = DGIAndGNN(dgi_model_simple_framework_only_degree,
                                                              gnn_model_downstream_simple_framework_only_degree, mlp_only_degree, False)
    optimizer_gnn_simple_framework_only_degree = torch.optim.Adam(
        gnn_model_simple_framework_only_degree.parameters(),
        lr=0.002322321872125943, weight_decay=4.234449204450303e-06)
    criterion_gnn_simple_framework_only_degree = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GIN and MLP----"""

    train_loader_gnn_model_simple_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_simple_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_simple_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
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
        num_layers=3,
        out_channels=512,
        norm=GraphNorm(256),
        dropout=0.2950022096499954,
        act='relu'
    )

    layer_sizes = [128 + 512] + [64] * 4 + [2]
    mlp = build_mlp(layer_sizes, nn.ELU, 0.3163058988828747)

    gnn_model_simple_framework_gin = DGIAndGNN(dgi_model_simple_framework_gin,
                                                              gnn_model_downstream_simple_framework_gin, mlp, False)
    optimizer_gnn_simple_framework_gin = torch.optim.Adam(
        gnn_model_simple_framework_gin.parameters(),
        lr=0.0004445808190363408, weight_decay=0.00010875565406402025)
    criterion_gnn_simple_framework_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS----"""

    train_loader_gnn_model_complex_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
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
        dropout=0.3935595942964136,
        act='gelu',
        aggr='max'
    )

    gnn_model_complex_framework = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework = torch.optim.Adam(
        gnn_model_complex_framework.parameters(),
        lr=0.001576586258891951, weight_decay=1.2813913397210403e-06)
    criterion_gnn_complex_framework = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS ONLY DEGREE----"""

    data_only_degree = copy.deepcopy(data)
    data_only_degree.topological_features = data_only_degree.topological_features[:, 0].unsqueeze(-1)

    train_loader_gnn_model_complex_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.train_mask
    )

    val_loader_gnn_model_complex_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.val_mask
    )

    test_loader_gnn_model_complex_framework_only_degree = NeighborLoader(
        data_only_degree,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=64,
        input_nodes=data_only_degree.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer_only_degree = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=32,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_only_degree.topological_features.shape[1],
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
        in_channels=data.num_features + 32,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
        dropout=0.21380223621527408,
        act='relu',
        aggr='mean',
        norm=GraphNorm(256)
    )

    gnn_model_complex_framework_only_degree = DGIPlusGNN(dgi_model_without_flipping_layer_only_degree,
                                                                gnn_model_downstream_framework_without_flipping_layer_only_degree,
                                                                False)
    optimizer_gnn_complex_framework_only_degree = torch.optim.Adam(
        gnn_model_complex_framework_only_degree.parameters(),
        lr=0.003249483526301171, weight_decay=1.0420379869125718e-06)
    criterion_gnn_complex_framework_only_degree = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS GIN----"""

    train_loader_gnn_model_complex_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.train_mask
    )

    val_loader_gnn_model_complex_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
        batch_size=64,
        input_nodes=data.val_mask
    )

    test_loader_gnn_model_complex_framework_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 40],
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
    gnn_model_downstream_framework_without_flipping_layer =  GIN(
        in_channels=data.num_features+128,
        hidden_channels=64,
        num_layers=3,
        out_channels=2,
        norm=GraphNorm(64),
        dropout=0.4685511903169562,
        act='leaky_relu'
    )

    gnn_model_complex_framework_gin = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework_gin = torch.optim.Adam(
        gnn_model_complex_framework_gin.parameters(),
        lr=0.0006401612564081927, weight_decay=3.869991153635051e-05)
    criterion_gnn_complex_framework_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GRAPHSAGE TOPOLOGICAL INPUT + DATA INPUT----"""

    data_all_features = copy.deepcopy(data)
    data_all_features.x = torch.cat([data.x, data.topological_features], dim=1)

    train_loader_gnn_all_features_graphsage = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=32,
        input_nodes=data_all_features.train_mask
    )

    val_loader_gnn_all_features_graphsage = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=32,
        input_nodes=data_all_features.val_mask
    )

    test_loader_gnn_all_features_graphsage = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[10, 10, 25],
        batch_size=32,
        input_nodes=data_all_features.test_mask
    )

    gnn_model_all_features_graphsage = GraphSAGE(
        in_channels=data_all_features.num_features,
        hidden_channels=128,
        num_layers=4,
        out_channels=2,
        dropout=0.2639459463249423,
        aggr='mean',
        act='elu'
    )
    optimizer_gnn_all_features_graphsage = torch.optim.Adam(gnn_model_all_features_graphsage.parameters(), lr=0.00085746256660606, weight_decay=1.995683035575948e-05)
    criterion_gnn_all_features_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GRAPHSAGE----"""

    train_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_simple_graphsage = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    gnn_model_simple_graphsage = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=4,
        out_channels=2,
        norm=BatchNorm(256),
        dropout=0.28082173362847057,
        aggr='max',
        act='leaky_relu',
    )
    optimizer_gnn_simple_graphsage = torch.optim.Adam(gnn_model_simple_graphsage.parameters(), lr=0.0018573604269558593, weight_decay=3.498293657545283e-06)
    criterion_gnn_simple_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GIN----"""

    train_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 30, 40],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 30, 40],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader_gnn_simple_gin = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 20, 30, 40],
        batch_size=32,
        input_nodes=data.test_mask
    )

    gnn_model_simple_gin = GIN(
        in_channels=data.num_features,
        hidden_channels=64,
        num_layers=2,
        out_channels=2,
        norm=BatchNorm(64),
        dropout=0.3986306469433091,
        act='relu'
    )

    optimizer_gnn_simple_gin = torch.optim.Adam(gnn_model_simple_gin.parameters(), lr=0.0013260343088210972
                                                , weight_decay=1.495957475057153e-05)
    criterion_gnn_simple_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GIN TOPOLOGICAL INPUT + DATA INPUT----"""

    data_all_features = copy.deepcopy(data)
    data_all_features.x = torch.cat([data.x, data.topological_features], dim=1)

    train_loader_gnn_all_features_gin = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[20, 20],
        batch_size=32,
        input_nodes=data_all_features.train_mask
    )

    val_loader_gnn_all_features_gin= NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[20, 20],
        batch_size=32,
        input_nodes=data_all_features.val_mask
    )

    test_loader_gnn_all_features_gin = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=[20, 20],
        batch_size=32,
        input_nodes=data_all_features.test_mask
    )

    gnn_model_all_features_gin = GIN(
        in_channels=data_all_features.num_features,
        hidden_channels=128,
        num_layers=3,
        out_channels=2,
        dropout=0.34784199594158705,
        act='leaky_relu'
    )

    optimizer_gnn_all_features_gin = torch.optim.Adam(gnn_model_all_features_gin.parameters(), lr=0.0010138466771643904
    , weight_decay=4.636489953583516e-06)
    criterion_gnn_all_features_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)


    """---------------------------------------------"""
    # Store all in a nested dict, all the models above must be in this dict
    model_dict = {

        'GraphSAGE + MLP': {
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
            'model': gnn_model_simple_framework,
            'optimizer': optimizer_gnn_simple_framework,
            'criterion': criterion_gnn_simple_framework,
            'train_set': train_loader_gnn_model_simple_framework,
            'val_set': val_loader_gnn_model_simple_framework,
            'test_set': test_loader_gnn_model_simple_framework
        },

        'simple_framework_without_flex_fronts_only_degree': {
            'model': gnn_model_simple_framework_only_degree,
            'optimizer': optimizer_gnn_simple_framework_only_degree,
            'criterion': criterion_gnn_simple_framework_only_degree,
            'train_set': train_loader_gnn_model_simple_framework_only_degree,
            'val_set': val_loader_gnn_model_simple_framework_only_degree,
            'test_set': test_loader_gnn_model_simple_framework_only_degree
        },

        'simple_framework_gin_without_flex_fronts': {
            'model': gnn_model_simple_framework_gin,
            'optimizer': optimizer_gnn_simple_framework_gin,
            'criterion': criterion_gnn_simple_framework_gin,
            'train_set': train_loader_gnn_model_simple_framework_gin,
            'val_set': val_loader_gnn_model_simple_framework_gin,
            'test_set': test_loader_gnn_model_simple_framework_gin
        },

        'complex_framework_without_flex_fronts': {
            'model': gnn_model_complex_framework,
            'optimizer': optimizer_gnn_complex_framework,
            'criterion': criterion_gnn_complex_framework,
            'train_set': train_loader_gnn_model_complex_framework,
            'val_set': val_loader_gnn_model_complex_framework,
            'test_set': test_loader_gnn_model_complex_framework
        },

        'complex_framework_without_flex_fronts_only_degree': {
            'model': gnn_model_complex_framework_only_degree,
            'optimizer': optimizer_gnn_complex_framework_only_degree,
            'criterion': criterion_gnn_complex_framework_only_degree,
            'train_set': train_loader_gnn_model_complex_framework_only_degree,
            'val_set': val_loader_gnn_model_complex_framework_only_degree,
            'test_set': test_loader_gnn_model_complex_framework_only_degree
        },

        'complex_framework_gin_without_flex_fronts': {
            'model': gnn_model_complex_framework_gin,
            'optimizer': optimizer_gnn_complex_framework_gin,
            'criterion': criterion_gnn_complex_framework_gin,
            'train_set': train_loader_gnn_model_complex_framework_gin,
            'val_set': val_loader_gnn_model_complex_framework_gin,
            'test_set': test_loader_gnn_model_complex_framework_gin
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

#change