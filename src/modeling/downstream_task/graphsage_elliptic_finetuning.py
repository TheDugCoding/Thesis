import optuna
import torch
import torch_geometric
from sklearn.metrics import f1_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE

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


#set dataset to use, hyperparameters and epochs
data = EllipticDataset(root=processed_data_path)
data = data[4]



def objective(trial):
    # Sample hyperparameters
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 3, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical("epochs", [5, 10, 15, 20])
    neighbours_size = trial.suggest_categorical("neighbours_size", [
        [10, 10],
        [20, 20],
        [10, 10, 25],
        [5, 5, 10],
        [15, 30],
        [10, 20, 40],
        [30, 50],
        [10, 20, 30, 40]
    ])

    # Define model
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=2,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=neighbours_size,
        batch_size=32,
        input_nodes=data.train_mask

    )

    val_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=neighbours_size,
        batch_size=32,
        input_nodes=data.val_mask
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

    def validate_once():
        model.eval()
        preds = []
        true = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                prob = torch.softmax(out[:batch.batch_size], dim=1)
                preds.append(prob.argmax(dim=1).cpu())
                true.append(batch.y[:batch.batch_size].cpu())
        preds = torch.cat(preds)
        true_labels = torch.cat(true)
        f1 = f1_score(true_labels, preds, average='weighted')
        return f1

    for _ in range(epochs):
        train_once()

    f1_val = validate_once()
    return f1_val


with open("graphsage_finetuning.txt", "w") as file:
    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    # Print and save the best trial
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write(f"  F1 Score: {trial.value}\n")
    file.write("  Best hyperparameters:\n")

    for key, value in trial.params.items():
        file.write(f"    {key}: {value}\n")