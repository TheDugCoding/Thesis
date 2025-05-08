import os

import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE

from src.data_preprocessing.preprocess import EllipticDataset
from src.modeling.utils.modeling_utils import train, validate, evaluate
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
epochs = 5

train_loader = NeighborLoader(
    data,
    shuffle=True,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data.train_mask

)

val_loader = NeighborLoader(
    data,
    shuffle=True,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data.val_mask
)

test_loader = NeighborLoader(
    data,
    shuffle=True,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes= data.test_mask
)

# Define model, optimizer, and loss function
model = GraphSAGE(
    in_channels=data.num_features,
    hidden_channels=256,
    num_layers=3,
    out_channels=2,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

#Run training and validation
with open("graphsage_training_log_losses_per_epoch.txt", "w") as file:
    for epoch in range(epochs):
        loss = train(train_loader, model, optimizer, device,
                                     criterion, False)
        log = f"Epoch {epoch+1:02d}, Loss: {loss:.6f}\n"
        print(log)
        file.write(log)
        accuracy, precision, recall, f1, auc_pr = validate(val_loader, model, device, False)

        # Logging
        log = (
            f"graphsage Metrics --- Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1: {f1:.4f}, AUC-PR: {auc_pr:.4f}\n"
        )
        print(log)
        file.write(log)


torch.save(model.state_dict(), os.path.join(trained_model_path, 'modeling_graphsage_trained.pth'))

print("\n----EVALUATION----\n")
with open(f"evaluation_performance_metrics_graphsage_trained.txt", "w") as f:
    f.write("----EVALUATION----\n")
    accuracy, precision, recall, f1, pr_auc, confusion_matrix_model, pr_auc_curve = evaluate(model, test_loader, device,
                                                                            'graphsage', False)
    f.write("----{graphsage}----\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"pr_auc Score: {pr_auc:.4f}\n")