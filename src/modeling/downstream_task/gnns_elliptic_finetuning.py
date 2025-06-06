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
from torch_geometric.nn import GraphSAGE, GAT, GIN

from src.data_preprocessing.preprocess import EllipticDataset
from src.modeling.downstream_task.dgi_and_mlp import DGIWithMLP, build_mlp
from src.modeling.downstream_task.graphsage_and_mlp import GraphsageWithMLP
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax_only_topological_features import \
    DeepGraphInfomaxWithoutFlexFronts, EncoderWithoutFlexFrontsGraphsage, corruption_without_flex_fronts
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/downstream_task/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
relative_path_finetuning_results = 'modeling/downstream_task/finetuning_results'
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)
finetuning_results = get_src_sub_folder(relative_path_finetuning_results)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

# set dataset to use, hyperparameters and epochs
data = EllipticDataset(root=processed_data_path)
data = data[1]

def get_norm(norm_type, hidden_channels):
    if norm_type == "batch":
        return BatchNorm(hidden_channels)
    elif norm_type == "layer":
        return LayerNorm(hidden_channels)
    elif norm_type == "graph":
        return GraphNorm(hidden_channels)
    else:
        return None

def objective_dgi_and_mlp(trial):
    # hyper-parameters
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    num_mlp_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
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

    activation_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU
    }

    activation_fn = activation_map[act]

    # Define MLP layers for classification
    layer_sizes = [128] + [hidden_channels] * num_mlp_layers + [2]
    mlp = build_mlp(layer_sizes, activation_fn, dropout)

    model = DGIWithMLP(dgi_model_dgi_and_mlp, mlp).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.train_mask

    )

    test_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.test_mask
    )

    def train_once():
        model.train()
        total_loss = 0
        total_examples = 0
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

    def test_once():
        model.eval()
        preds = []
        true = []
        probs = []
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

    for _ in range(int(epochs)):
        train_once()

    pr_auc = test_once()
    return pr_auc

def objective_graphsage_and_mlp(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    aggr = trial.suggest_categorical("aggr", ["mean", "sum", "max"])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    output_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    output_channels_mlp = trial.suggest_categorical("output_channels_mlp", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
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

    norm_layer = get_norm(norm_choice, hidden_channels)

    # Define model
    gnn_model_graphsage_and_mlp = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=output_channels,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    # Define MLP layers for classification
    mlp = nn.Sequential(
        nn.Linear(output_channels, output_channels_mlp),
        nn.ReLU(),
        nn.Linear(output_channels_mlp, 2),
    )

    model = GraphsageWithMLP(gnn_model_graphsage_and_mlp, mlp).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.train_mask

    )

    test_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.test_mask
    )

    def train_once():
        model.train()
        total_loss = 0
        total_examples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
        return total_loss / total_examples

    def test_once():
        model.eval()
        preds = []
        true = []
        probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
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

    for _ in range(int(epochs)):
        train_once()

    pr_auc = test_once()
    return pr_auc

def objective_graphsage(trial):
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

    norm_layer = get_norm(norm_choice, hidden_channels)

    # Define model
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.train_mask

    )

    test_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.test_mask
    )

    def train_once():
        model.train()
        total_loss = 0
        total_examples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
        return total_loss / total_examples

    def test_once():
        model.eval()
        preds = []
        true = []
        probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
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

    for _ in range(int(epochs)):
        train_once()

    pr_auc = test_once()
    return pr_auc

def objective_graphsage_all_features(trial):
    data_all_features = copy.deepcopy(data)
    data_all_features.x = torch.cat([data.x, data.topological_features], dim=1)

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

    norm_layer = get_norm(norm_choice, hidden_channels)

    # Define model
    model = GraphSAGE(
        in_channels=data_all_features.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
        act=act,
        aggr=aggr,
        norm=norm_layer).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data_all_features.train_mask

    )

    test_loader = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data_all_features.test_mask
    )

    def train_once():
        model.train()
        total_loss = 0
        total_examples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
        return total_loss / total_examples

    def test_once():
        model.eval()
        preds = []
        true = []
        probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
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

    for _ in range(int(epochs)):
        train_once()

    pr_auc = test_once()
    return pr_auc

def objective_gat(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
    act = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    heads = trial.suggest_categorical("heads", [1, 2, 4, 8, 16])
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

    norm_layer = get_norm(norm_choice, hidden_channels)

    # Define model
    model = GAT(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
        heads=heads,
        v2=True,
        act=act,
        norm=norm_layer).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.train_mask

    )

    test_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.test_mask
    )

    def train_once():
        model.train()
        total_loss = 0
        total_examples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
        return total_loss / total_examples

    def test_once():
        model.eval()
        preds = []
        true = []
        probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
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

    for _ in range(int(epochs)):
        train_once()

    pr_auc = test_once()
    return pr_auc

def objective_gin(trial):
    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
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

    norm_layer = get_norm(norm_choice, hidden_channels)

    # Define model
    model = GIN(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
        act=act,
        norm=norm_layer).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.train_mask

    )

    test_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data.test_mask
    )

    def train_once():
        model.train()
        total_loss = 0
        total_examples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
        return total_loss / total_examples

    def test_once():
        model.eval()
        preds = []
        true = []
        probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
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

    for _ in range(int(epochs)):
        train_once()

    pr_auc = test_once()
    return pr_auc

def objective_gin_all_features(trial):
    data_all_features = copy.deepcopy(data)
    data_all_features.x = torch.cat([data.x, data.topological_features], dim=1)

    # hyper-parameters
    norm_choice = trial.suggest_categorical("norm", ["batch", "layer", "graph", None])
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

    norm_layer = get_norm(norm_choice, hidden_channels)

    # Define model
    model = GIN(
        in_channels=data_all_features.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
        act=act,
        norm=norm_layer).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data_all_features.train_mask

    )

    test_loader = NeighborLoader(
        data_all_features,
        shuffle=True,
        num_neighbors=ast.literal_eval(neighbours_size),
        batch_size=32,
        input_nodes=data_all_features.test_mask
    )

    def train_once():
        model.train()
        total_loss = 0
        total_examples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
        return total_loss / total_examples

    def test_once():
        model.eval()
        preds = []
        true = []
        probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
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

    for _ in range(int(epochs)):
        train_once()

    pr_auc = test_once()
    return pr_auc


# with open(os.path.join(finetuning_results, "graphsage_and_mlp_finetuning.txt"), "w") as file:
#     # run Optuna study
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective_graphsage_and_mlp, n_trials=30, show_progress_bar=True)
#
#     # print and save the best trial
#     file.write("Best trial:\n")
#     trial = study.best_trial
#     file.write(f"  PR-AUC Score: {trial.value}\n")
#     file.write("  Best hyperparameters:\n")
#
#     for key, value in trial.params.items():
#         file.write(f"    {key}: {value}\n")

# with open(os.path.join(finetuning_results, "dgi_and_mlp_finetuning.txt"), "w") as file:
#     # run Optuna study
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective_dgi_and_mlp, n_trials=30, show_progress_bar=True)
#
#     # print and save the best trial
#     file.write("Best trial:\n")
#     trial = study.best_trial
#     file.write(f"  PR-AUC Score: {trial.value}\n")
#     file.write("  Best hyperparameters:\n")
#
#     for key, value in trial.params.items():
#         file.write(f"    {key}: {value}\n")

# with open(os.path.join(finetuning_results, "graphsage_finetuning.txt"), "w") as file:
#     # run Optuna study
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective_graphsage, n_trials=30, show_progress_bar=True)
#
#     # print and save the best trial
#     file.write("Best trial:\n")
#     trial = study.best_trial
#     file.write(f"  PR-AUC Score: {trial.value}\n")
#     file.write("  Best hyperparameters:\n")
#
#     for key, value in trial.params.items():
#         file.write(f"    {key}: {value}\n")

# with open(os.path.join(finetuning_results, "graphsage_all_features_finetuning.txt"), "w") as file:
#     # run Optuna study
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective_graphsage_all_features, n_trials=30, show_progress_bar=True)
#
#     # print and save the best trial
#     file.write("Best trial:\n")
#     trial = study.best_trial
#     file.write(f"  PR-AUC Score: {trial.value}\n")
#     file.write("  Best hyperparameters:\n")
#
#     for key, value in trial.params.items():
#         file.write(f"    {key}: {value}\n")

with open(os.path.join(finetuning_results, "gat_finetuning.txt"), "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_gat, n_trials=30, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

# with open(os.path.join(finetuning_results, "gin_finetuning.txt"), "w") as file:
#     # run Optuna study
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective_gin, n_trials=30, show_progress_bar=True)
#
#     # print and save the best trial
#     file.write("Best trial:\n")
#     trial = study.best_trial
#     file.write(f"  PR-AUC Score: {trial.value}\n")
#     file.write("  Best hyperparameters:\n")
#
#     for key, value in trial.params.items():
#         file.write(f"    {key}: {value}\n")

# with open(os.path.join(finetuning_results, "gin_finetuning_all_features.txt"), "w") as file:
#     # run Optuna study
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective_gin_all_features, n_trials=30, show_progress_bar=True)
#
#     # print and save the best trial
#     file.write("Best trial:\n")
#     trial = study.best_trial
#     file.write(f"  PR-AUC Score: {trial.value}\n")
#     file.write("  Best hyperparameters:\n")
#
#     for key, value in trial.params.items():
#         file.write(f"    {key}: {value}\n")
