import ast

import optuna
import torch
from torch_geometric.loader import NeighborLoader
from torch import nn
import os

from src.data_preprocessing.preprocess import RealDataTraining
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax_only_topological_features import DeepGraphInfomaxWithoutFlexFronts, EncoderWithoutFlexFrontsGraphsage, corruption_without_flex_fronts, train, EncoderWithoutFlexFrontsGIN

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/pre_training/topological_pre_training/trained_models'
relative_path_finetuning_results = 'modeling/pre_training/topological_pre_training/finetuning_results'
processed_data_path = get_data_sub_folder(relative_path_processed)
finetuning_results = get_src_sub_folder(relative_path_finetuning_results)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = RealDataTraining(root=processed_data_path)

def objective(trial):
    # Suggest hyperparameters
    num_layers = trial.suggest_int("num_layers", 2, 4)
    act_name = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256, 512])
    output_channels = trial.suggest_categorical('output_channels', [32, 64, 128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    neighbours_size = trial.suggest_categorical("neighbours_size", [
        "[10, 10]", "[20, 20]", "[15, 30]", "[30, 50]",
        "[5, 5, 10]", "[10, 10, 25]", "[10, 20, 40]",
        "[5, 5, 5, 10]", "[10, 10, 10, 25]", "[10, 15, 20, 40]"
    ])

    parsed = ast.literal_eval(neighbours_size)
    if len(parsed) != num_layers:
        raise optuna.TrialPruned()

    activation_map = {
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "gelu": torch.nn.GELU,
    }

    activation_fn = activation_map[act_name]

    # Load data
    data_rabo = dataset[0].clone()
    data_ethereum = dataset[1].clone()
    data_stable_20 = dataset[2].clone()

    # x contains a dummy feature, replace it with only topological features
    data_rabo.x = data_rabo.topological_features
    data_ethereum.x = data_ethereum.topological_features
    data_stable_20.x = data_stable_20.topological_features

    # Set up loaders
    train_loader_rabo = NeighborLoader(data_rabo, batch_size=batch_size, shuffle=True, num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_ethereum = NeighborLoader(data_ethereum, batch_size=batch_size, shuffle=True,
                                           num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_stable_20 = NeighborLoader(data_stable_20, batch_size=batch_size, shuffle=True,
                                            num_neighbors=ast.literal_eval(neighbours_size))

    train_loaders = [train_loader_rabo, train_loader_ethereum, train_loader_stable_20]

    # Define model and optimizer
    model = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=output_channels,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_rabo.num_features,
                                                  hidden_channels=hidden_channels,
                                                  output_channels=output_channels,
                                                  layers=num_layers,
                                                  activation_fn=activation_fn
                                                  ),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(5):
        loss = train(epoch, train_loaders, model, optimizer, 'BCEdgi')
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return loss

def objective_infonce(trial):
    # Suggest hyperparameters
    num_layers = trial.suggest_int("num_layers", 2, 4)
    act_name = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256, 512])
    output_channels = trial.suggest_categorical('output_channels', [32, 64, 128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    neighbours_size = trial.suggest_categorical("neighbours_size", [
        "[10, 10]", "[20, 20]", "[15, 30]", "[30, 50]",
        "[5, 5, 10]", "[10, 10, 25]", "[10, 20, 40]",
        "[5, 5, 5, 10]", "[10, 10, 10, 25]", "[10, 15, 20, 40]"
    ])

    parsed = ast.literal_eval(neighbours_size)
    if len(parsed) != num_layers:
        raise optuna.TrialPruned()

    activation_map = {
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "gelu": torch.nn.GELU,
    }

    activation_fn = activation_map[act_name]

    # Load data
    data_rabo = dataset[0].clone()
    data_ethereum = dataset[1].clone()
    data_stable_20 = dataset[2].clone()

    # x contains a dummy feature, replace it with only topological features
    data_rabo.x = data_rabo.topological_features
    data_ethereum.x = data_ethereum.topological_features
    data_stable_20.x = data_stable_20.topological_features

    # Set up loaders
    train_loader_rabo = NeighborLoader(data_rabo, batch_size=batch_size, shuffle=True,
                                       num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_ethereum = NeighborLoader(data_ethereum, batch_size=batch_size, shuffle=True,
                                           num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_stable_20 = NeighborLoader(data_stable_20, batch_size=batch_size, shuffle=True,
                                            num_neighbors=ast.literal_eval(neighbours_size))

    train_loaders = [train_loader_rabo, train_loader_ethereum, train_loader_stable_20]

    # Define model and optimizer
    model = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=output_channels,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_rabo.num_features,
                                                  hidden_channels=hidden_channels,
                                                  output_channels=output_channels,
                                                  layers=num_layers,
                                                  activation_fn=activation_fn
                                                  ),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(5):
        loss = train(epoch, train_loaders, model, optimizer, 'infoNCE')
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return loss

def objective_gin(trial):
    # Suggest hyperparameters
    num_layers = trial.suggest_int("num_layers", 2, 4)
    act_name = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256, 512])
    output_channels = trial.suggest_categorical('output_channels', [32, 64, 128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    neighbours_size = trial.suggest_categorical("neighbours_size", [
        "[10, 10]", "[20, 20]", "[15, 30]", "[30, 50]",
        "[5, 5, 10]", "[10, 10, 25]", "[10, 20, 40]",
        "[5, 5, 5, 10]", "[10, 10, 10, 25]", "[10, 15, 20, 40]"
    ])

    parsed = ast.literal_eval(neighbours_size)
    if len(parsed) != num_layers:
        raise optuna.TrialPruned()

    activation_map = {
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "gelu": torch.nn.GELU,
    }

    activation_fn = activation_map[act_name]

    # Load data
    data_rabo = dataset[0].clone()
    data_ethereum = dataset[1].clone()
    data_stable_20 = dataset[2].clone()

    # x contains a dummy feature, replace it with only topological features
    data_rabo.x = data_rabo.topological_features
    data_ethereum.x = data_ethereum.topological_features
    data_stable_20.x = data_stable_20.topological_features

    # Set up loaders
    train_loader_rabo = NeighborLoader(data_rabo, batch_size=batch_size, shuffle=True, num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_ethereum = NeighborLoader(data_ethereum, batch_size=batch_size, shuffle=True,
                                           num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_stable_20 = NeighborLoader(data_stable_20, batch_size=batch_size, shuffle=True,
                                            num_neighbors=ast.literal_eval(neighbours_size))

    train_loaders = [train_loader_rabo, train_loader_ethereum, train_loader_stable_20]

    # Define model and optimizer
    model = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=output_channels,
        encoder=EncoderWithoutFlexFrontsGIN(
            input_channels=data_rabo.num_features,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            layers=num_layers,
            activation_fn=activation_fn
        ),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(5):
        loss = train(epoch, train_loaders, model, optimizer, 'BCEdgi')
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return loss

def objective_only_degree(trial):
    # Suggest hyperparameters
    num_layers = trial.suggest_int("num_layers", 2, 4)
    act_name = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256, 512])
    output_channels = trial.suggest_categorical('output_channels', [32, 64, 128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    neighbours_size = trial.suggest_categorical("neighbours_size", [
        "[10, 10]", "[20, 20]", "[15, 30]", "[30, 50]",
        "[5, 5, 10]", "[10, 10, 25]", "[10, 20, 40]",
        "[5, 5, 5, 10]", "[10, 10, 10, 25]", "[10, 15, 20, 40]"
    ])

    parsed = ast.literal_eval(neighbours_size)
    if len(parsed) != num_layers:
        raise optuna.TrialPruned()

    activation_map = {
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "gelu": torch.nn.GELU,
    }

    activation_fn = activation_map[act_name]

    # Load data
    data_rabo = dataset[0].clone()
    data_ethereum = dataset[1].clone()
    data_stable_20 = dataset[2].clone()

    # x contains a dummy feature, replace it with only topological features
    data_rabo.x = data_rabo.topological_features[:, 0].unsqueeze(-1)
    data_ethereum.x = data_ethereum.topological_features[:, 0].unsqueeze(-1)
    data_stable_20.x = data_stable_20.topological_features[:, 0].unsqueeze(-1)

    # Set up loaders
    train_loader_rabo = NeighborLoader(data_rabo, batch_size=batch_size, shuffle=True,
                                       num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_ethereum = NeighborLoader(data_ethereum, batch_size=batch_size, shuffle=True,
                                           num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_stable_20 = NeighborLoader(data_stable_20, batch_size=batch_size, shuffle=True,
                                            num_neighbors=ast.literal_eval(neighbours_size))

    train_loaders = [train_loader_rabo, train_loader_ethereum, train_loader_stable_20]

    # Define model and optimizer
    model = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=output_channels,
        encoder=EncoderWithoutFlexFrontsGraphsage(input_channels=data_rabo.num_features,
                                                  hidden_channels=hidden_channels,
                                                  output_channels=output_channels,
                                                  layers=num_layers,
                                                  activation_fn=activation_fn
                                                  ),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_without_flex_fronts).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(5):
        loss = train(epoch, train_loaders, model, optimizer, 'BCEdgi')
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return loss

if __name__ == '__main__':

    # with open(os.path.join(finetuning_results, "deep_graph_infomax_with_topological_features_rabo_ecr_20_ethereum_finetuning.txt"), "w") as file:
    #     # run Optuna study
    #     study = optuna.create_study(
    #         direction='minimize',
    #         pruner=optuna.pruners.MedianPruner()
    #     )
    #     study.optimize(objective, n_trials=60)
    #
    #     # print and save the best trial
    #     file.write("Best trial:\n")
    #     trial = study.best_trial
    #     print("Best trial:")
    #     print(f"  Loss: {trial.value}")
    #     file.write(f"  Loss: {trial.value}\n")
    #     file.write("  Best hyperparameters:\n")
    #     for key, value in trial.params.items():
    #         file.write(f"    {key}: {value}\n")

    with open(os.path.join(finetuning_results, "deep_graph_infomax_infonce_with_topological_features_rabo_ecr_20_ethereum_finetuning.txt"), "w") as file:
        # run Optuna study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective_infonce, n_trials=60)

        # print and save the best trial
        file.write("Best trial:\n")
        trial = study.best_trial
        print("Best trial:")
        print(f"  Loss: {trial.value}")
        file.write(f"  Loss: {trial.value}\n")
        file.write("  Best hyperparameters:\n")
        for key, value in trial.params.items():
            file.write(f"    {key}: {value}\n")

    # with open(os.path.join(finetuning_results,"deep_graph_infomax_gin_with_topological_feature_rabo_ethereum_ecr20.txt"), "w") as file:
    #     # run Optuna study
    #     study = optuna.create_study(
    #     #         direction='minimize',
    #     #         pruner=optuna.pruners.MedianPruner()
    #     #     )
    #     #     study.optimize(objective_gin, n_trials=60)
    #
    #     # print and save the best trial
    #     file.write("Best trial:\n")
    #     trial = study.best_trial
    #     print("Best trial:")
    #     print(f"  Loss: {trial.value}")
    #     file.write(f"  Loss: {trial.value}\n")
    #     file.write("  Best hyperparameters:\n")
    #     for key, value in trial.params.items():
    #         file.write(f"    {key}: {value}\n")

    # with open(os.path.join(finetuning_results, "deep_graph_infomax_with_topological_features_rabo_ecr_20_ethereum_finetuning_only_degree.txt"), "w") as file:
    #
    #     study = optuna.create_study(
    #         direction='minimize',
    #         pruner=optuna.pruners.MedianPruner()
    #     )
    #     study.optimize(objective_only_degree, n_trials=60)
    #
    #     # print and save the best trial
    #     file.write("Best trial:\n")
    #     trial = study.best_trial
    #     print("Best trial:")
    #     print(f"  Loss: {trial.value}")
    #     file.write(f"  Loss: {trial.value}\n")
    #     file.write("  Best hyperparameters:\n")
    #     for key, value in trial.params.items():
    #         file.write(f"    {key}: {value}\n")