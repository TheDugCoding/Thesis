import copy
import os

import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm
from torch_geometric.nn import GraphSAGE, GIN

from src.modeling.downstream_task.dgi_and_mlp import DGIWithMLP
from src.modeling.downstream_task.dgi_and_mlp import build_mlp
from src.modeling.downstream_task.graphsage_and_mlp import GraphsageWithMLP
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


def parse_finetuning_file(filepath):
    """Parse a finetuning result file and return the best hyperparameters as a dict, or None if file is empty/missing."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        content = f.read().strip()
    if not content:
        return None
    params = {}
    in_params = False
    for line in content.split('\n'):
        line = line.strip()
        if line == 'Best hyperparameters:':
            in_params = True
            continue
        if in_params and ':' in line:
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                params[key] = [int(x.strip()) for x in value[1:-1].split(',')]
            elif value == 'None':
                params[key] = None
            else:
                try:
                    params[key] = int(value)
                except ValueError:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value
    return params if params else None


def make_norm(norm_str, channels):
    """Convert a norm string to a PyG norm module."""
    if norm_str is None:
        return None
    if norm_str == 'batch':
        return BatchNorm(channels)
    elif norm_str == 'layer':
        return LayerNorm(channels)
    elif norm_str == 'graph':
        return GraphNorm(channels)
    return None


_ACT_CLS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'selu': nn.SELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
}


def act_str_to_cls(act_str):
    """Convert an activation string to a PyTorch activation class."""
    return _ACT_CLS.get(act_str, nn.ReLU)


    #define here the models to test against the framework

def model_list_rq1_ex1(data, data_only_degree, data_all_features):
    """
    :param data: the dataset that is used
    :return: a dict containing all the gnns to test against the framework
    """

    downstream_ft_path = get_src_sub_folder('modeling/downstream_task/finetuning_results')
    framework_ft_path = get_src_sub_folder('modeling/final_framework/finetuning_results')

    # DGI output dimensions fixed by pre-trained model architecture
    DGI_OUTPUT_FULL = 128
    DGI_OUTPUT_DEGREE = 32

    #list of models to test
    """----Graphsage and MLP----"""
    p = parse_finetuning_file(os.path.join(downstream_ft_path, 'graphsage_and_mlp_finetuning.txt'))
    data_gnn_model_graphsage_and_mlp = data
    num_neighbors_gnn_model_graphsage_and_mlp = p['neighbours_size']
    batch_size_gnn_model_graphsage_and_mlp = p['batch_size']

    hidden_ch = p['hidden_channels']
    out_ch = p['output_channels']
    out_ch_mlp = p['output_channels_mlp']

    gnn_model_graphsage_and_mlp = GraphSAGE(
        in_channels=data[0].num_features,
        hidden_channels=hidden_ch,
        num_layers=p['num_layers'],
        out_channels=out_ch,
        norm=make_norm(p['norm'], hidden_ch),
        dropout=p['dropout'],
        aggr=p['aggr'],
        act=p['act'],
    )

    # Define MLP layers for classification
    mlp = nn.Sequential(
        nn.Linear(out_ch, out_ch_mlp),
        nn.ReLU(),
        nn.Linear(out_ch_mlp, 2),
    )

    gnn_model_graphsage_and_mlp = GraphsageWithMLP(gnn_model_graphsage_and_mlp, mlp)
    optimizer_gnn_model_graphsage_and_mlp = torch.optim.Adam(
        gnn_model_graphsage_and_mlp.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_model_graphsage_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # list of models  to test
    """----DGI and MLP----"""
    p = parse_finetuning_file(os.path.join(downstream_ft_path, 'dgi_and_mlp_finetuning.txt'))
    data_gnn_model_dgi_and_mlp = data
    num_neighbors_gnn_model_dgi_and_mlp = [10, 20, 40]
    batch_size_gnn_model_dgi_and_mlp = 32

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_dgi_and_mlp = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=DGI_OUTPUT_FULL, encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1], hidden_channels=DGI_OUTPUT_FULL, output_channels=DGI_OUTPUT_FULL, layers=4, activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_dgi_and_mlp.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_dgi_and_mlp.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    hidden_ch = p['hidden_channels']
    num_layers_mlp = p['num_layers']
    layer_sizes = [DGI_OUTPUT_FULL] + [hidden_ch] * num_layers_mlp + [2]
    mlp_dgi_and_mlp = build_mlp(layer_sizes, act_str_to_cls(p['act']), p['dropout'])

    gnn_model_model_dgi_and_mlp = DGIWithMLP(dgi_model_dgi_and_mlp, mlp_dgi_and_mlp)
    optimizer_gnn_model_dgi_and_mlp = torch.optim.Adam(
        gnn_model_model_dgi_and_mlp.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_model_dgi_and_mlp = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----PAF, DGI, GRAPHSAGE and MLP----"""
    p = parse_finetuning_file(os.path.join(framework_ft_path, 'framework_simple_finetuning_free_neighbor.txt'))
    data_gnn_model_simple_framework = data
    num_neighbors_gnn_model_simple_framework = p['neighbours_size']
    batch_size_gnn_model_simple_framework = p['batch_size']

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=DGI_OUTPUT_FULL,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1],
                                                  hidden_channels=DGI_OUTPUT_FULL, output_channels=DGI_OUTPUT_FULL, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_simple_framework.load_state_dict(torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_simple_framework.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    hidden_ch = p['hidden_channels']
    out_ch = p['output_channels']
    gnn_model_downstream_simple_framework = GraphSAGE(
        in_channels=data[0].num_features,
        hidden_channels=hidden_ch,
        num_layers=p['num_layers'],
        out_channels=out_ch,
        dropout=p['dropout'],
        act=p['act'],
        aggr=p['aggr'],
        norm=make_norm(p['norm'], hidden_ch)
    )

    hidden_ch_mlp = p['hidden_channels_mlp']
    num_layers_mlp = p['num_layers_mlp']
    layer_sizes = [DGI_OUTPUT_FULL + out_ch] + [hidden_ch_mlp] * num_layers_mlp + [2]
    mlp_simple_framework = build_mlp(layer_sizes, act_str_to_cls(p['act_mlp']), p['dropout_mlp'])

    gnn_model_simple_framework = DGIAndGNN(dgi_model_simple_framework, gnn_model_downstream_simple_framework, mlp_simple_framework, False)
    optimizer_gnn_simple_framework = torch.optim.Adam(
        gnn_model_simple_framework.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_simple_framework = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GRAPHSAGE and MLP ONLY DEGREE----"""
    p = parse_finetuning_file(os.path.join(framework_ft_path, 'framework_simple_finetuning_only_degree_dgi.txt'))
    data_gnn_model_simple_framework_only_degree = data_only_degree
    num_neighbors_gnn_model_simple_framework_only_degree = [10, 10, 25]
    batch_size_gnn_model_simple_framework_only_degree = p['batch_size']

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework_only_degree = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=DGI_OUTPUT_DEGREE,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_gnn_model_simple_framework_only_degree[0].topological_features.shape[1],
                                                  hidden_channels=64, output_channels=DGI_OUTPUT_DEGREE, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_simple_framework_only_degree.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_GraphSage_no_flex_front_only_topo_rabo_ecr_20_only_degree.pth')))

    for layer in dgi_model_simple_framework_only_degree.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    hidden_ch = p['hidden_channels']
    out_ch = p['output_channels']
    gnn_model_downstream_simple_framework_only_degree = GraphSAGE(
        in_channels=data[0].num_features,
        hidden_channels=hidden_ch,
        num_layers=3,
        out_channels=out_ch,
        dropout=p['dropout'],
        act=p['act'],
        aggr=p['aggr'],
        norm=make_norm(p['norm'], hidden_ch)
    )

    hidden_ch_mlp = p['hidden_channels_mlp']
    num_layers_mlp = p['num_layers_mlp']
    layer_sizes = [DGI_OUTPUT_DEGREE + out_ch] + [hidden_ch_mlp] * num_layers_mlp + [2]
    mlp_only_degree = build_mlp(layer_sizes, act_str_to_cls(p['act_mlp']), p['dropout_mlp'])

    gnn_model_simple_framework_only_degree = DGIAndGNN(dgi_model_simple_framework_only_degree,
                                                              gnn_model_downstream_simple_framework_only_degree, mlp_only_degree, False)
    optimizer_gnn_simple_framework_only_degree = torch.optim.Adam(
        gnn_model_simple_framework_only_degree.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_simple_framework_only_degree = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----SIMPLE FRAMEWORK DGI, GIN and MLP----"""
    p = parse_finetuning_file(os.path.join(framework_ft_path, 'framework_simple_finetuning_gin.txt'))
    data_gnn_model_simple_framework_gin = data
    num_neighbors_gnn_model_simple_framework_gin = [10, 20, 40]
    batch_size_gnn_model_simple_framework_gin = p['batch_size']

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_simple_framework_gin = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=DGI_OUTPUT_FULL,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1],
                                                  hidden_channels=DGI_OUTPUT_FULL, output_channels=DGI_OUTPUT_FULL, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_simple_framework_gin.load_state_dict(torch.load(
        os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_simple_framework_gin.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    hidden_ch = p['hidden_channels']
    out_ch = p['output_channels']
    gnn_model_downstream_simple_framework_gin = GIN(
        in_channels=data[0].num_features,
        hidden_channels=hidden_ch,
        num_layers=3,
        out_channels=out_ch,
        norm=make_norm(p['norm'], hidden_ch),
        dropout=p['dropout'],
        act=p['act']
    )

    hidden_ch_mlp = p['hidden_channels_mlp']
    num_layers_mlp = p['num_layers_mlp']
    layer_sizes = [DGI_OUTPUT_FULL + out_ch] + [hidden_ch_mlp] * num_layers_mlp + [2]
    mlp = build_mlp(layer_sizes, act_str_to_cls(p['act_mlp']), p['dropout_mlp'])

    gnn_model_simple_framework_gin = DGIAndGNN(dgi_model_simple_framework_gin,
                                                              gnn_model_downstream_simple_framework_gin, mlp, False)
    optimizer_gnn_simple_framework_gin = torch.optim.Adam(
        gnn_model_simple_framework_gin.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_simple_framework_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS----"""
    p = parse_finetuning_file(os.path.join(framework_ft_path, 'framework_complex_finetuning_free_neighbor.txt'))
    data_gnn_model_complex_framework = data
    num_neighbors_gnn_model_complex_framework = p['neighbours_size']
    batch_size_gnn_model_complex_framework = p['batch_size']

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=DGI_OUTPUT_FULL,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1],
                                                  hidden_channels=DGI_OUTPUT_FULL, output_channels=DGI_OUTPUT_FULL, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(os.path.join(trained_dgi_model_path, 'modeling_dgi_no_flex_front_only_topo_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_without_flipping_layer.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    hidden_ch = p['hidden_channels']
    gnn_model_downstream_framework_without_flipping_layer = GraphSAGE(
        in_channels=data[0].num_features + DGI_OUTPUT_FULL,
        hidden_channels=hidden_ch,
        num_layers=p['num_layers'],
        out_channels=2,
        dropout=p['dropout'],
        act=p['act'],
        aggr=p['aggr'],
        norm=make_norm(p['norm'], hidden_ch),
    )

    gnn_model_complex_framework = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework = torch.optim.Adam(
        gnn_model_complex_framework.parameters(),
        lr=p['lr'] if p else 0.0001781660288878494,
        weight_decay=p['weight_decay'] if p else 0.00048693914641231314)
    criterion_gnn_complex_framework = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS ONLY DEGREE----"""
    p = parse_finetuning_file(os.path.join(framework_ft_path, 'framework_complex_finetuning_only_degree_dgi.txt'))
    data_gnn_model_complex_framework_only_degree = data_only_degree
    num_neighbors_gnn_model_complex_framework_only_degree = [10, 10, 25]
    batch_size_gnn_model_complex_framework_only_degree = p['batch_size']

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer_only_degree = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=DGI_OUTPUT_DEGREE,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_gnn_model_complex_framework_only_degree[0].topological_features.shape[1],
                                                  hidden_channels=64, output_channels=DGI_OUTPUT_DEGREE, layers=4,
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

    hidden_ch = p['hidden_channels']
    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer_only_degree = GraphSAGE(
        in_channels=data[0].num_features + DGI_OUTPUT_DEGREE,
        hidden_channels=hidden_ch,
        num_layers=3,
        out_channels=2,
        dropout=p['dropout'],
        act=p['act'],
        aggr=p['aggr'],
        norm=make_norm(p['norm'], hidden_ch),
    )

    gnn_model_complex_framework_only_degree = DGIPlusGNN(dgi_model_without_flipping_layer_only_degree,
                                                                gnn_model_downstream_framework_without_flipping_layer_only_degree,
                                                                False)
    optimizer_gnn_complex_framework_only_degree = torch.optim.Adam(
        gnn_model_complex_framework_only_degree.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_complex_framework_only_degree = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----COMPLEX FRAMEWORK WITHOUT FLEX FRONTS GIN----"""
    p = parse_finetuning_file(os.path.join(framework_ft_path, 'framework_complex_finetuning_gin.txt'))
    data_gnn_model_complex_framework_gin = data
    num_neighbors_gnn_model_complex_framework_gin = [10, 20, 40]
    batch_size_gnn_model_complex_framework_gin = 64

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=DGI_OUTPUT_FULL,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1],
                                                  hidden_channels=DGI_OUTPUT_FULL, output_channels=DGI_OUTPUT_FULL, layers=4,
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

    hidden_ch = p['hidden_channels']
    # same model as in graphsage_elliptic, used in the framework
    gnn_model_downstream_framework_without_flipping_layer = GIN(
        in_channels=data[0].num_features + DGI_OUTPUT_FULL,
        hidden_channels=hidden_ch,
        num_layers=3,
        out_channels=2,
        norm=make_norm(p['norm'], hidden_ch),
        dropout=p['dropout'],
        act=p['act']
    )

    gnn_model_complex_framework_gin = DGIPlusGNN(dgi_model_without_flipping_layer,
                                                                gnn_model_downstream_framework_without_flipping_layer,
                                                                False)
    optimizer_gnn_complex_framework_gin = torch.optim.Adam(
        gnn_model_complex_framework_gin.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_complex_framework_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GRAPHSAGE TOPOLOGICAL INPUT + DATA INPUT----"""
    p = parse_finetuning_file(os.path.join(downstream_ft_path, 'graphsage_all_features_finetuning.txt'))
    data_gnn_all_features_graphsage = data_all_features
    num_neighbors_gnn_all_features_graphsage = p['neighbours_size']
    batch_size_gnn_all_features_graphsage = 32

    hidden_ch = p['hidden_channels']
    gnn_model_all_features_graphsage = GraphSAGE(
        in_channels=data_gnn_all_features_graphsage[0].num_features,
        hidden_channels=hidden_ch,
        num_layers=p['num_layers'],
        out_channels=2,
        dropout=p['dropout'],
        aggr=p['aggr'],
        act=p['act'],
        norm=make_norm(p['norm'], hidden_ch),
    )
    optimizer_gnn_all_features_graphsage = torch.optim.Adam(
        gnn_model_all_features_graphsage.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_all_features_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GRAPHSAGE----"""
    p = parse_finetuning_file(os.path.join(downstream_ft_path, 'graphsage_finetuning.txt'))
    data_gnn_simple_graphsage = data
    num_neighbors_gnn_simple_graphsage = p['neighbours_size']
    batch_size_gnn_simple_graphsage = 32

    hidden_ch = p['hidden_channels']
    gnn_model_simple_graphsage = GraphSAGE(
        in_channels=data[0].num_features,
        hidden_channels=hidden_ch,
        num_layers=p['num_layers'],
        out_channels=2,
        norm=make_norm(p['norm'], hidden_ch),
        dropout=p['dropout'],
        aggr=p['aggr'],
        act=p['act'],
    )
    optimizer_gnn_simple_graphsage = torch.optim.Adam(
        gnn_model_simple_graphsage.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_simple_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)



    """----GIN----"""
    p = parse_finetuning_file(os.path.join(downstream_ft_path, 'gin_finetuning.txt'))
    data_gnn_simple_gin = data
    num_neighbors_gnn_simple_gin = p['neighbours_size']
    batch_size_gnn_simple_gin = 32

    hidden_ch = p['hidden_channels']
    gnn_model_simple_gin = GIN(
        in_channels=data[0].num_features,
        hidden_channels=hidden_ch,
        num_layers=p['num_layers'],
        out_channels=2,
        dropout=p['dropout'],
        act=p['act'],
        norm=make_norm(p['norm'], hidden_ch),
    )

    optimizer_gnn_simple_gin = torch.optim.Adam(
        gnn_model_simple_gin.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
    criterion_gnn_simple_gin = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GIN TOPOLOGICAL INPUT + DATA INPUT----"""
    p = parse_finetuning_file(os.path.join(downstream_ft_path, 'gin_finetuning_all_features.txt'))
    data_gnn_all_features_gin = data_all_features
    num_neighbors_gnn_all_features_gin = p['neighbours_size']
    batch_size_gnn_all_features_gin = 32

    hidden_ch = p['hidden_channels']
    gnn_model_all_features_gin = GIN(
        in_channels=data_gnn_all_features_gin[0].num_features,
        hidden_channels=hidden_ch,
        num_layers=p['num_layers'],
        out_channels=2,
        norm=make_norm(p['norm'], hidden_ch),
        dropout=p['dropout'],
        act=p['act'],
    )

    optimizer_gnn_all_features_gin = torch.optim.Adam(
        gnn_model_all_features_gin.parameters(),
        lr=p['lr'],
        weight_decay=p['weight_decay'])
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