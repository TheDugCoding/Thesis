import ast

import optuna
import torch
import torch_geometric
from sklearn.metrics import average_precision_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE,GAT, GIN

from src.data_preprocessing.preprocess import EllipticDataset
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/downstream_task/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

# set dataset to use, hyperparameters and epochs
data = EllipticDataset(root=processed_data_path)
data = data[4]


def objective_graphsage(trial):
    # hyper-parameters
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

    # Define model
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
        act=act).to(device)

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
        probs_class0 = probs[:, 0]
        pr_auc = average_precision_score(true_labels, probs_class0, pos_label=0)
        return pr_auc

    for _ in range(epochs):
        train_once()

    pr_auc = test_once()
    return pr_auc

def objective_gat(trial):
    # hyper-parameters
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

    # Define model
    model = GAT(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
        heads=heads,
        act=act).to(device)

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
        probs_class0 = probs[:, 0]
        pr_auc = average_precision_score(true_labels, probs_class0, pos_label=0)
        return pr_auc

    for _ in range(epochs):
        train_once()

    pr_auc = test_once()
    return pr_auc

def objective_gin(trial):
    # hyper-parameters
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

    # Define model
    model = GIN(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
        act=act).to(device)

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
        probs_class0 = probs[:, 0]
        pr_auc = average_precision_score(true_labels, probs_class0, pos_label=0)
        return pr_auc

    for _ in range(epochs):
        train_once()

    pr_auc = test_once()
    return pr_auc

with open("graphsage_finetuning.txt", "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_graphsage, n_trials=30, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")

with open("gat_finetuning.txt", "w") as file:
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

with open("gin_finetuning.txt", "w") as file:
    # run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_gin, n_trials=30, show_progress_bar=True)

    # print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  PR-AUC Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")