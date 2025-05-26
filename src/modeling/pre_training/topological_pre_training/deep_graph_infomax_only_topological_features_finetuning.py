import ast

import optuna
import torch
from torch_geometric.loader import NeighborLoader
from torch import nn

from src.data_preprocessing.preprocess import RealDataTraining
from src.utils import get_data_folder, get_data_sub_folder
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax_only_topological_features import DeepGraphInfomaxWithoutFlexFronts, EncoderWithoutFlexFrontsGraphsage, corruption_without_flex_fronts, train

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def objective(trial):
    # Suggest hyperparameters
    num_layers = trial.suggest_int("num_layers", 2, 4)
    act_name = trial.suggest_categorical("act", ["relu", "leaky_relu", "elu", "gelu"])
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    output_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    neighbours_size = trial.suggest_categorical("neighbours_size", [
        "[10, 10]",
        "[20, 20]",
        "[15, 30]",
        "[30, 50]",
        "[5, 5, 10]",
        "[10, 10, 25]",
        "[10, 20, 40]",
    ])

    activation_map = {
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "gelu": torch.nn.GELU,
    }

    activation_fn = activation_map[act_name]

    # Load data
    dataset = RealDataTraining(root=processed_data_path)
    data_rabo, data_ethereum, data_stable_20 = dataset[0], dataset[1], dataset[2]

    # x contains a dummy feature, replace it with only topological features
    data_rabo.x = data_rabo.topological_features
    data_ethereum.x = data_ethereum.topological_features
    data_stable_20.x = data_stable_20.topological_features

    # Set up loaders
    train_loader_rabo = NeighborLoader(data_rabo, batch_size=64, shuffle=True, num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_ethereum = NeighborLoader(data_ethereum, batch_size=64, shuffle=True,
                                           num_neighbors=ast.literal_eval(neighbours_size))
    train_loader_stable_20 = NeighborLoader(data_stable_20, batch_size=64, shuffle=True,
                                            num_neighbors=ast.literal_eval(neighbours_size))

    train_loaders = [train_loader_rabo]

    # Define model and optimizer
    model = DeepGraphInfomaxWithoutFlexFronts(
        hidden_channels=hidden_channels,
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
        loss = train(epoch, train_loaders, model, optimizer)

    return loss

if __name__ == '__main__':

    with open("deep_graph_infomax_without_topological_features_only_rabo_finetuning.txt", "w") as file:
        # run Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)

        # print and save the best trial
        file.write("Best trial:\n")
        trial = study.best_trial
        print("Best trial:")
        print(f"  Loss: {trial.value}")
        file.write(f"  Loss: {trial.value}\n")
        file.write("  Best hyperparameters:\n")
        for key, value in trial.params.items():
            file.write(f"    {key}: {value}\n")