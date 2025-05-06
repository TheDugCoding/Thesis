import os

import matplotlib.pyplot as plt
import torch
import torch_geometric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from models_to_test import model_list

from src.data_preprocessing.preprocess import EllipticDataset
from src.modeling.final_framework.framework import DGIPlusGNN, corruption
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax import DeepGraphInfomax, Encoder
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder
from src.modeling.utils.modeling_utils import train, validate, evaluate
from src.modeling.testing.models_to_test import model_list

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/testing/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

# Load your dataset
data = EllipticDataset(root=processed_data_path)

data = data[4]
epochs = 5

train_loader = NeighborLoader(
    data,
    shuffle=False,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data.train_mask
)

val_loader = NeighborLoader(
    data,
    shuffle=False,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data.val_mask
)

test_loader = NeighborLoader(
    data,
    shuffle=False,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data.test_mask
)



#model to test set the device
models_to_compare = model_list(data)
for name, components in models_to_compare.items():
    components['model'] = components['model'].to(device)

if not os.path.exists(os.path.join(trained_model_path, 'final_framework_trained.pth')):
    # Run training
    with open("training_log_per_epoch.txt", "w") as file:
        for epoch in range(epochs):

            #train the framework
            log = (f"\n\n---TRAINING--- Epoch {epoch + 1:02d}\n")
            print(log)
            file.write(log)
            #train the models, framework need a special variable
            for name, components in models_to_compare.items():
                if name == 'framework':
                    loss_gnn = train(train_loader, components['model'], components['optimizer'], device, components['criterion'], True)
                else:
                    loss_gnn = train(train_loader, components['model'], components['optimizer'], device,
                                     components['criterion'], False)
                log = (f"Loss {name}: {loss_gnn:.6f}\n")
                print(log)
                file.write(log)

            #validation
            log = (
                f"---VALIDATION--- Epoch {epoch + 1:02d}\n")
            file.write(log)

            for name, components in models_to_compare.items():
                if name == 'framework':
                    accuracy_gnn, recall_gnn, f1_gnn, auc_pr_gnn = validate(val_loader, components['model'], device, True)
                else:
                    accuracy_gnn, recall_gnn, f1_gnn, auc_pr_gnn = validate(val_loader, components['model'], device,
                                                                            False)
                # Logging
                log = (
                    f"{name} Metrics --- Accuracy: {accuracy_gnn:.4f}, Recall: {recall_gnn:.4f}, "
                    f"F1: {f1_gnn:.4f}, AUC-PR: {auc_pr_gnn:.4f}\n"
                )
                print(log)
                file.write(log)

    for name, components in models_to_compare.items():
        torch.save(components['model'].state_dict(), os.path.join(trained_model_path, f'{name}_gnn_trained.pth'))
else:
    for name, components in models_to_compare.items():
        components['model'].load_state_dict(
            torch.load(os.path.join(trained_model_path, f'{name}_gnn_trained.pth'), map_location=device))

# Inference
print("\n----EVALUATION----\n")
with open(f"evaluation_performance_metrics_trained.txt", "w") as f:
    f.write("----EVALUATION----\n")
    for name, components in models_to_compare.items():
        if name == 'framework':
            accuracy, recall, f1, pr_auc, confusion_matrix_model = evaluate(components['model'], test_loader, device, name, True)
        else:
            accuracy, recall, f1, pr_auc, confusion_matrix_model = evaluate(components['model'], test_loader, device,
                                                                            name, False)
        f.write(f"----{name}----\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"pr_auc Score: {pr_auc:.4f}\n")
