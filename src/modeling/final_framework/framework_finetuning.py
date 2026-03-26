import ast
import copy
import os

import optuna
import torch
import torch.nn as nn
import torch_geometric
from sklearn.metrics import average_precision_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm
from torch_geometric.nn import GraphSAGE, GIN

from src.data_preprocessing.preprocess import EllipticDataset
from src.modeling.downstream_task.dgi_and_mlp import build_mlp
from src.modeling.final_framework.framework_complex import DGIPlusGNN
from src.modeling.final_framework.framework_simple import DGIAndGNN
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax_only_topological_features import \
    DeepGraphInfomaxWithoutFlexFronts, EncoderWithoutFlexFrontsGraphsage, corruption_without_flex_fronts, \
    EncoderWithoutFlexFrontsGIN
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder
from torch_geometric.data import Batch

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/downstream_task/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
relative_path_finetuning_results = 'modeling/final_framework/finetuning_results'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)
finetuning_results = get_src_sub_folder(relative_path_finetuning_results)

NUMBER_OF_TRIALS = 60

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

# set dataset tokk use, hyperparameters and epochs
original_data = EllipticDataset(root=processed_data_path)
data = original_data

def reduce_train_val_masks(dataset, n_train, n_val, train_range=(0, 29), val_range=(29, 36)):
    """
    Reduces train_mask across training graphs and val_mask across validation graphs
    in a multi-graph dataset, with balanced class sampling.

    Args:
        dataset: list/dataset of PyG Data objects (42 graphs)
        n_train: total number of training nodes to keep (across all training graphs)
        n_val: total number of validation nodes to keep (across all val graphs)
        train_range: (start, end) indices for training graphs
        val_range: (start, end) indices for validation graphs

    Returns:
        modified dataset (deep copy) with reduced masks
    """
    dataset_copy = copy.deepcopy(dataset)

    def reduce_mask_across_graphs(graphs, mask_attr, n_total):
        # Collect all candidate indices as (graph_idx, node_idx, label)
        candidates_per_class = {}
        for g_idx, g in enumerate(graphs):
            mask = getattr(g, mask_attr)
            y = g.y
            for node_idx in mask.nonzero(as_tuple=True)[0].tolist():
                label = y[node_idx].item()
                if label not in candidates_per_class:
                    candidates_per_class[label] = []
                candidates_per_class[label].append((g_idx, node_idx))

        num_classes = len(candidates_per_class)
        per_class = n_total // num_classes

        # Zero out all masks first
        for g in graphs:
            getattr(g, mask_attr).fill_(False)

        # Sample balanced and set masks
        for label, candidates in candidates_per_class.items():
            perm = torch.randperm(len(candidates))
            selected = perm[:per_class]
            for s in selected:
                g_idx, node_idx = candidates[s.item()]
                getattr(graphs[g_idx], mask_attr)[node_idx] = True

    # Reduce train masks across training graphs
    train_graphs = [dataset_copy[i] for i in range(train_range[0], train_range[1])]
    reduce_mask_across_graphs(train_graphs, 'train_mask', n_train)


    return dataset_copy


def train_once(model, data, neighbours_size, batch_size, optimizer, criterion):
    model.train()
    total_loss = 0
    total_examples = 0

    batched_data = Batch.from_data_list(data)

    train_loader = NeighborLoader(
        batched_data,
        shuffle=True,
        num_neighbors=neighbours_size,
        batch_size=batch_size,
        input_nodes=batched_data.train_mask

    )

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        total_examples += batch.batch_size
    return total_loss / total_examples


def test_once(model, data, neighbours_size, batch_size):
    model.eval()
    preds = []
    true = []
    probs = []

    batched_data = Batch.from_data_list(data)

    test_loader = NeighborLoader(
        batched_data,
        shuffle=True,
        num_neighbors=neighbours_size,
        batch_size=batch_size,
        input_nodes=batched_data.test_mask
    )

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            prob = torch.softmax(out[:batch.batch_size], dim=1)
            preds.append(prob.argmax(dim=1).cpu())
            probs.append(prob.cpu())
            true.append(batch.y[:batch.batch_size].cpu())

    true_labels = torch.cat(true)
    # PR-AUC
    probs = torch.cat(probs, dim=0)
    probs_class0 = probs[:, 0]
    pr_auc = average_precision_score(true_labels, probs_class0, pos_label=0, average='weighted')
    return pr_auc


def get_norm(norm_type, hidden_channels):
    if norm_type == "batch":
        return BatchNorm(hidden_channels)
    elif norm_type == "layer":
        return LayerNorm(hidden_channels)
    elif norm_type == "graph":
        return GraphNorm(hidden_channels)
    else:
        return None


def objective_framework_complex(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    aggr = trial.suggest_categorical("aggr", ["mean", "sum", "max"])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])
    neighbours_size = trial.suggest_categorical("neighbours_size", [
        "[10, 10]",
        "[20, 20]",
        "[15, 30]",
        "[30, 50]",
        "[5, 5, 10]",
        "[10, 10, 25]",
        "[10, 20, 40]",
        "[10, 20, 30, 40]"
    ])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    norm_layer = get_norm(norm_choice, hidden_channels)

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

    # Define model
    graphsage = GraphSAGE(
        in_channels=data[0].num_features + 128,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    model = DGIPlusGNN(dgi_model_without_flipping_layer, graphsage, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data[0:29], ast.literal_eval(neighbours_size), batch_size, optimizer, criterion)

    pr_auc = test_once(model, data[36:42], ast.literal_eval(neighbours_size), batch_size)
    return pr_auc


def objective_framework_complex_only_degree_dgi(trial):
    # hyper-parameters
    data_only_degree = []
    for d in data:
        d_copy = copy.deepcopy(d)
        d_copy.topological_features = d_copy.topological_features[:, 0].unsqueeze(-1)
        data_only_degree.append(d_copy)
    

    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    aggr = trial.suggest_categorical("aggr", ["mean", "sum", "max"])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    norm_layer = get_norm(norm_choice, hidden_channels)

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=32,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_only_degree[0].topological_features.shape[1],
                                                  hidden_channels=64, output_channels=32, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path,
                         'modeling_dgi_GraphSage_no_flex_front_only_topo_rabo_ecr_20_only_degree.pth')))

    for layer in dgi_model_without_flipping_layer.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Define model
    graphsage = GraphSAGE(
        in_channels=data_only_degree[0].num_features + 32,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=2,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    model = DGIPlusGNN(dgi_model_without_flipping_layer, graphsage, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data_only_degree[0:29], [10, 10, 25], batch_size, optimizer, criterion)

    pr_auc = test_once(model, data_only_degree[36:42], [10, 10, 25], batch_size)
    return pr_auc


def objective_framework_complex_gin_encoder(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    aggr = trial.suggest_categorical("aggr", ["mean", "sum", "max"])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    norm_layer = get_norm(norm_choice, hidden_channels)

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=512,
        encoder=EncoderWithoutFlexFrontsGIN(input_channels=data[0].topological_features.shape[1],
                                            hidden_channels=128, output_channels=512, layers=4,
                                            activation_fn=torch.nn.GELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path, 'modeling_dgi_GIN_rabo_ethereum_erc_20.pth')))

    for layer in dgi_model_without_flipping_layer.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Define model
    graphsage = GraphSAGE(
        in_channels=data[0].num_features + 512,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=2,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    model = DGIPlusGNN(dgi_model_without_flipping_layer, graphsage, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data[0:29], [10, 10, 25], batch_size, optimizer, criterion)

    pr_auc = test_once(model, data[36:42], [10, 10, 25], batch_size)
    return pr_auc


def objective_framework_complex_infonce(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    aggr = trial.suggest_categorical("aggr", ["mean", "sum", "max"])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    norm_layer = get_norm(norm_choice, hidden_channels)

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=64,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data[0].topological_features.shape[1],
                                                  hidden_channels=64, output_channels=64, layers=4,
                                                  activation_fn=torch.nn.ReLU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path,
                         'modeling_dgi_GraphSage_no_flex_front_only_topo_rabo_ecr_20_infonce.pth')))

    for layer in dgi_model_without_flipping_layer.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Define model
    graphsage = GraphSAGE(
        in_channels=data[0].num_features + 64,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=2,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    model = DGIPlusGNN(dgi_model_without_flipping_layer, graphsage, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data[0:29], [10, 20, 40], batch_size, optimizer, criterion)

    pr_auc = test_once(model, data[36:42], [10, 20, 40], batch_size)
    return pr_auc


def objective_framework_complex_gin(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])

    norm_layer = get_norm(norm_choice, hidden_channels)

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

    # Define model
    gin = GIN(
        in_channels=data[0].num_features + 128,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=2,
        dropout=dropout,
        act=act,
        norm=norm_layer).to(device)

    model = DGIPlusGNN(dgi_model_without_flipping_layer, gin, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data[0:29], [10, 20, 40], 64, optimizer, criterion)

    pr_auc = test_once(model, data[36:42], [10, 20, 40], 64)
    return pr_auc


def objective_framework_complex_first_layer_unfreeze(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    aggr = trial.suggest_categorical("aggr", ["mean", "sum", "max"])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    norm_layer = get_norm(norm_choice, hidden_channels)

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

    skip_first_layer = True
    for idx, layer in enumerate(dgi_model_without_flipping_layer.encoder.layers):
        if idx == 0 and skip_first_layer:
            continue  # Skip the first layer
        for param in layer.parameters():
            param.requires_grad = False

    # Define model
    graphsage = GraphSAGE(
        in_channels=data[0].num_features + 128,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=2,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    model = DGIPlusGNN(dgi_model_without_flipping_layer, graphsage, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data[0:29], [10, 20, 40], batch_size, optimizer, criterion)

    pr_auc = test_once(model, data[36:42], [10, 20, 40], batch_size)
    return pr_auc


def objective_framework_complex_last_layer_unfreeze(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    aggr = trial.suggest_categorical("aggr", ["mean", "sum", "max"])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    norm_layer = get_norm(norm_choice, hidden_channels)

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

    # freeze all the layers except the last one
    for idx in range(len(dgi_model_without_flipping_layer.encoder.layers) - 1):
        for param in dgi_model_without_flipping_layer.encoder.layers[idx].parameters():
            param.requires_grad = False

    # Define model
    graphsage = GraphSAGE(
        in_channels=data[0].num_features + 128,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=2,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    model = DGIPlusGNN(dgi_model_without_flipping_layer, graphsage, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data[0:29], [10, 20, 40], batch_size, optimizer, criterion)

    pr_auc = test_once(model, data[36:42], [10, 20, 40], batch_size)
    return pr_auc


def objective_framework_simple(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    aggr = trial.suggest_categorical("aggr", ["mean", "sum", "max"])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    act_mlp = trial.suggest_categorical("act_mlp", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    output_channels = trial.suggest_categorical("output_channels", [128, 256, 512])
    hidden_channels_mlp = trial.suggest_categorical("hidden_channels_mlp", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    num_mlp_layers = trial.suggest_int("num_layers_mlp", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    dropout_mlp = trial.suggest_float("dropout_mlp", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])
    neighbours_size = trial.suggest_categorical("neighbours_size", [
        "[10, 10]",
        "[20, 20]",
        "[15, 30]",
        "[30, 50]",
        "[5, 5, 10]",
        "[10, 10, 25]",
        "[10, 20, 40]",
        "[10, 20, 30, 40]"
    ])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    activation_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU
    }

    norm_layer = get_norm(norm_choice, hidden_channels)

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

    # Define model
    graphsage = GraphSAGE(
        in_channels=data[0].num_features,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=output_channels,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    activation_fn = activation_map[act_mlp]

    # Define MLP layers for classification
    layer_sizes = [128 + output_channels] + [hidden_channels_mlp] * num_mlp_layers + [2]
    mlp = build_mlp(layer_sizes, activation_fn, dropout_mlp)

    model = DGIAndGNN(dgi_model_without_flipping_layer, graphsage, mlp, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data[0:29], ast.literal_eval(neighbours_size), batch_size, optimizer, criterion)

    pr_auc = test_once(model, data[36:42], ast.literal_eval(neighbours_size), batch_size)
    return pr_auc


def objective_framework_simple_only_degree_dgi(trial):
    # hyper-parameters
    data_only_degree = []
    for d in data:
        d_copy = copy.deepcopy(d)
        d_copy.topological_features = d_copy.topological_features[:, 0].unsqueeze(-1)
        data_only_degree.append(d_copy)

    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    aggr = trial.suggest_categorical("aggr", ["mean", "sum", "max"])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    act_mlp = trial.suggest_categorical("act_mlp", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    output_channels = trial.suggest_categorical("output_channels", [128, 256, 512])
    hidden_channels_mlp = trial.suggest_categorical("hidden_channels_mlp", [64, 128, 256])
    num_mlp_layers = trial.suggest_int("num_layers_mlp", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    dropout_mlp = trial.suggest_float("dropout_mlp", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    activation_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU
    }

    norm_layer = get_norm(norm_choice, hidden_channels)

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model_without_flipping_layer = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=32,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_only_degree[0].topological_features.shape[1],
                                                  hidden_channels=64, output_channels=32, layers=4,
                                                  activation_fn=torch.nn.ELU),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts)
    # load the pretrained parameters
    dgi_model_without_flipping_layer.load_state_dict(
        torch.load(
            os.path.join(trained_dgi_model_path,
                         'modeling_dgi_GraphSage_no_flex_front_only_topo_rabo_ecr_20_only_degree.pth')))

    for layer in dgi_model_without_flipping_layer.encoder.layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Define model
    graphsage = GraphSAGE(
        in_channels=data_only_degree[0].num_features,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=output_channels,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    activation_fn = activation_map[act_mlp]

    # Define MLP layers for classification
    layer_sizes = [32 + output_channels] + [hidden_channels_mlp] * num_mlp_layers + [2]
    mlp = build_mlp(layer_sizes, activation_fn, dropout_mlp)

    model = DGIAndGNN(dgi_model_without_flipping_layer, graphsage, mlp, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data_only_degree[0:29], [10, 10, 25], batch_size, optimizer, criterion)

    pr_auc = test_once(model, data_only_degree[36:42], [10, 10, 25], batch_size)
    return pr_auc


def objective_framework_simple_gin(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    act_mlp = trial.suggest_categorical("act_mlp", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    output_channels = trial.suggest_categorical("output_channels", [128, 256, 512])
    hidden_channels_mlp = trial.suggest_categorical("hidden_channels_mlp", [64, 128, 256])
    num_mlp_layers = trial.suggest_int("num_layers_mlp", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    dropout_mlp = trial.suggest_float("dropout_mlp", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20, 50])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    activation_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU
    }

    norm_layer = get_norm(norm_choice, hidden_channels)

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

    # Define model
    gin = GIN(
        in_channels=data[0].num_features,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=output_channels,
        dropout=dropout,
        act=act,
        norm=norm_layer).to(device)

    activation_fn = activation_map[act_mlp]

    # Define MLP layers for classification
    layer_sizes = [128 + output_channels] + [hidden_channels_mlp] * num_mlp_layers + [2]
    mlp = build_mlp(layer_sizes, activation_fn, dropout_mlp)

    model = DGIAndGNN(dgi_model_without_flipping_layer, gin, mlp, False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(int(epochs)):
        train_once(model, data[0:29], [10, 20, 40], batch_size, optimizer, criterion)

    pr_auc = test_once(model, data[36:42], [10, 20, 40], batch_size)
    return pr_auc


with open(os.path.join(finetuning_results, "framework_complex_finetuning_free_neighbor.txt"), "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_framework_complex, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

with open(os.path.join(finetuning_results, "framework_complex_first_layer_unfreeze.txt"), "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_framework_complex_first_layer_unfreeze, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

with open(os.path.join(finetuning_results, "framework_complex_last_layer_unfreeze.txt"), "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_framework_complex_last_layer_unfreeze, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

with open(os.path.join(finetuning_results, "framework_simple_finetuning_free_neighbor.txt"), "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_framework_simple, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

"""----GIN VARIATION---"""
with open(os.path.join(finetuning_results, "framework_complex_finetuning_gin.txt"), "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_framework_complex_gin, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

with open(os.path.join(finetuning_results, "framework_simple_finetuning_gin.txt"), "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_framework_simple_gin, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

"""-------only degree finetuning rq1"""
with open(os.path.join(finetuning_results, "framework_complex_finetuning_only_degree_dgi.txt"), "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_framework_complex_only_degree_dgi, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

with open(os.path.join(finetuning_results, "framework_simple_finetuning_only_degree_dgi.txt"), "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_framework_simple_only_degree_dgi, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

# '''------------------answering rq2---------------------'''
#
# """----GIN ENCODER VARIATION---"""
# with open(os.path.join(finetuning_results, "framework_complex_gin_encoder_finetuning.txt"), "w") as file:
#     # run Optuna study
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective_framework_complex_gin_encoder, n_trials=30, show_progress_bar=True)
#
#     # print and save the best trial
#     file.write("Best trial:\n")
#     trial = study.best_trial
#     file.write(f"  PR-AUC Score: {trial.value}\n")
#     file.write("  Best hyperparameters:\n")
#
#     for key, value in trial.params.items():
#         file.write(f"    {key}: {value}\n")
#
#     '''-----answering rq3'''
#
# """----INFONCE VARIATION---"""
#
# train_set_sizes = [20, 100, 500, 1000, 2000, 5000]
#
# for train_set_size in train_set_sizes:
#
#     # Create a reduced copy each time — don't overwrite the original `data`
#     data = reduce_train_val_masks(original_data, train_set_size, 300)
#
#     print('--------------------')
#     total_train = sum(data[i].train_mask.sum().item() for i in range(29))
#     print(f"Train set size: {total_train}")
#     print('--------------------')
#
#     #
#     with open(os.path.join(finetuning_results, f"framework_simple_finetuning_train_set_size_{train_set_size}.txt"),
#               "w") as file:
#         # run Optuna study
#         study = optuna.create_study(direction="maximize")
#         study.optimize(objective_framework_simple, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)
#
#         # print and save the best trial
#         file.write("Best trial:\n")
#         trial = study.best_trial
#         file.write(f"  PR-AUC Score: {trial.value}\n")
#         file.write("  Best hyperparameters:\n")
#
#         for key, value in trial.params.items():
#             file.write(f"    {key}: {value}\n")
#
#     with open(os.path.join(finetuning_results, f"framework_complex_finetuning_train_set_size_{train_set_size}.txt"),
#               "w") as file:
#         # run Optuna study
#         study = optuna.create_study(direction="maximize")
#         study.optimize(objective_framework_complex, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)
#
#         # print and save the best trial
#         file.write("Best trial:\n")
#         trial = study.best_trial
#         file.write(f"  PR-AUC Score: {trial.value}\n")
#         file.write("  Best hyperparameters:\n")
#
#         for key, value in trial.params.items():
#             file.write(f"    {key}: {value}\n")
#
#     with open(os.path.join(finetuning_results, f"framework_complex_finetuning_gin_train_set_size_{train_set_size}.txt"),
#               "w") as file:
#         # run Optuna study
#         study = optuna.create_study(direction="maximize")
#         study.optimize(objective_framework_complex_gin, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)
#
#         # print and save the best trial
#         file.write("Best trial:\n")
#         trial = study.best_trial
#         file.write(f"  PR-AUC Score: {trial.value}\n")
#         file.write("  Best hyperparameters:\n")
#
#         for key, value in trial.params.items():
#             file.write(f"    {key}: {value}\n")
#
#     with open(os.path.join(finetuning_results, f"framework_simple_finetuning_gin_train_set_size_{train_set_size}.txt"),
#               "w") as file:
#         # run Optuna study
#         study = optuna.create_study(direction="maximize")
#         study.optimize(objective_framework_simple_gin, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)
#
#         # print and save the best trial
#         file.write("Best trial:\n")
#         trial = study.best_trial
#         file.write(f"  PR-AUC Score: {trial.value}\n")
#         file.write("  Best hyperparameters:\n")
#
#         for key, value in trial.params.items():
#             file.write(f"    {key}: {value}\n")
#     with open(os.path.join(finetuning_results, "framework_complex_encoder_finetuning_infonce.txt"), "w") as file:
#         # run Optuna study
#         study = optuna.create_study(direction="maximize")
#         study.optimize(objective_framework_complex_infonce, n_trials=NUMBER_OF_TRIALS, show_progress_bar=True)
#
#         # print and save the best trial
#         file.write("Best trial:\n")
#         trial = study.best_trial
#         file.write(f"  PR-AUC Score: {trial.value}\n")
#         file.write("  Best hyperparameters:\n")
#
#         for key, value in trial.params.items():
#             file.write(f"    {key}: {value}\n")
#
# # change
