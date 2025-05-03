import os

import matplotlib.pyplot as plt
import torch
import torch_geometric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from src.data_preprocessing.preprocess import EllipticDataset
from src.modeling.final_framework.framework import DGIPlusGNN, corruption
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax import DeepGraphInfomax, Encoder
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder
from src.modeling.utils.modeling_utils import train, validate, evaluate

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

# define the framework, first DGI and then the GNN used in the downstream task
dgi_model = DeepGraphInfomax(
    hidden_channels=64, encoder=Encoder(64, 64, 2),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
# load the pretrained parameters
dgi_model.load_state_dict(torch.load(os.path.join(trained_dgi_model_path, 'modeling_graphsage_unsup_trained.pth')))
# reset first layer, be sure that the hidden channels are the same in DGI
dgi_model.encoder.dataset_convs[0] = SAGEConv(4, 64)
# freeze layers 2 and 3, let layer 1 learn
for param in dgi_model.encoder.conv2.parameters():
    param.requires_grad = False
for param in dgi_model.encoder.conv3.parameters():
    param.requires_grad = False

    # define the downstream GNN, it is the same as graphsage elliptic++. However the input is different according to the framework architecture
    # gnn_model = GAT(data.num_features + 64, 64, 3,
    #            8).to(device)

# same model as in graphsage_elliptic
gnn_model_downstream_framework = GraphSAGE(
    in_channels=data.num_features + 64,
    hidden_channels=256,
    num_layers=3,
    out_channels=2,
).to(device)

#model to test
gnn_model_graphsage = GraphSAGE(
    in_channels=data.num_features,
    hidden_channels=256,
    num_layers=3,
    out_channels=2,
).to(device)

model_framework = DGIPlusGNN(dgi_model, gnn_model_downstream_framework).to(device)
optimizer_framework = torch.optim.Adam(model_framework.parameters(), lr=0.005, weight_decay=5e-4)
optimizer_graphsage = torch.optim.Adam(gnn_model_graphsage.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

if not os.path.exists(os.path.join(trained_model_path, 'final_framework_trained.pth')):
    # Run training
    with open("training_log_per_epoch.txt", "w") as file:
        for epoch in range(5):
            loss_framework = train(train_loader, model_framework, optimizer_framework, device, criterion, True)
            loss_graphsage = train(train_loader, gnn_model_graphsage, optimizer_graphsage, device, criterion, False)
            log = f"Epoch {epoch + 1:02d}, Loss Framework: {loss_framework:.6f}\n, Loss Graphsage GNN: {loss_graphsage:.6f}\n"
            accuracy_framework, recall_framework, f1_framework, auc_pr_framework = validate(val_loader, model_framework, device, True)
            accuracy_graphsage, recall_graphsage, f1_graphsage, auc_pr_graphsage = validate(val_loader, gnn_model_graphsage, device, False)

            # Logging
            log = (
                f"Epoch {epoch + 1:02d}\n"
                f"  Loss Framework     : {loss_framework:.6f}\n"
                f"  Loss Graphsage GNN    : {loss_graphsage:.6f}\n"
                f"  Framework Metrics  - Accuracy: {accuracy_framework:.4f}, Recall: {recall_framework:.4f}, "
                f"F1: {f1_framework:.4f}, AUC-PR: {auc_pr_framework:.4f}\n"
                f"  Graphsage Metrics   - Accuracy: {accuracy_graphsage:.4f}, Recall: {recall_graphsage:.4f}, "
                f"F1: {f1_graphsage:.4f}, AUC-PR: {auc_pr_graphsage:.4f}\n\n"
            )

            print(log)
            file.write(log)

    torch.save(model_framework.state_dict(), os.path.join(trained_model_path, 'final_framework_trained.pth'))
    torch.save(gnn_model_graphsage.state_dict(), os.path.join(trained_model_path, 'graphsage_gnn_trained.pth'))
else:
    model_framework.load_state_dict(
        torch.load(os.path.join(trained_model_path, 'final_framework_trained.pth'), map_location=device))
    gnn_model_graphsage.load_state_dict(
        torch.load(os.path.join(trained_model_path, 'graphsage_gnn_trained.pth'), map_location=device))

print("\n----EVALUATION----\n")
# Inference
evaluate(model_framework, test_loader, device, 'framework', True)
evaluate(gnn_model_graphsage, test_loader, device, 'graphsage', False)
