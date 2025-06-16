import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,f1_score, recall_score, average_precision_score
from torch_geometric.nn import GATConv
from torch_geometric.utils import subgraph
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

from src.data_preprocessing.preprocess import EllipticDataset, AmlSimDataset
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder
from src.modeling.utils.modeling_utils import train, validate, evaluate

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

def reduce_train_val_masks(data, n_train, n_val):
    """
    Creates two balanced masks (train and val) from the original train_mask,
    with no overlap between them.

    Args:
        data: PyG Data object with .train_mask and .y
        n_train: total number of training samples to keep
        n_val: total number of validation samples to keep

    Returns:
        train_mask_new: Boolean mask with n_train True values, balanced across classes
        val_mask_new: Boolean mask with n_val True values, balanced across classes
    """
    y = data.y
    train_mask = data.train_mask

    unique_classes = torch.unique(y[train_mask])
    num_classes = len(unique_classes)

    train_per_class = n_train // num_classes
    val_per_class = n_val // num_classes

    train_mask_new = torch.zeros_like(train_mask, dtype=torch.bool)
    val_mask_new = torch.zeros_like(train_mask, dtype=torch.bool)

    for c in unique_classes:
        # indices of class c in original train_mask
        idx = (y == c) & train_mask
        class_indices = idx.nonzero(as_tuple=True)[0]

        # shuffle indices
        shuffled = class_indices[torch.randperm(len(class_indices))]

        # split into train and val, no overlap
        train_indices = shuffled[:train_per_class]
        val_indices = shuffled[train_per_class:train_per_class + val_per_class]

        train_mask_new[train_indices] = True
        val_mask_new[val_indices] = True

    return train_mask_new, val_mask_new

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False)

    def forward(self, x, edge_index):
        #Dropout helps prevent overfitting by randomly nullifying outputs from neurons during the training process. This encourages the network to learn redundant representations for everything and hence, increases the model's ability to generalize.
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Training loop
def train_test(data):
    model.train()
    data = data.to(device)  # Ensure subgraph is on the correct device

    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Compute loss on all nodes
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

    return float(loss.item())

def validate(val_loader):
    model.eval()
    preds = []
    true = []
    probs = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            prob = torch.softmax(out[:batch.batch_size], dim=1)
            preds.append(prob.argmax(dim=1).cpu())
            probs.append(prob.cpu())
            true.append(batch.y[:batch.batch_size].cpu())

    preds = torch.cat(preds)
    probs = torch.cat(probs)
    true_labels = torch.cat(true)

    accuracy = (preds == true_labels).sum().item() / true_labels.size(0)
    recall = recall_score(true_labels, preds, average='macro')
    f1 = f1_score(true_labels, preds, average='weighted')

    # AUC-PR
    probs_class0 = probs[:, 0]
    pr_auc = average_precision_score(true_labels, probs_class0, pos_label=0, average='weighted')

    print(f"Accuracy: {accuracy:.4f}, Recall (macro): {recall:.4f}, F1 Score: {f1:.4f}, AUC-PR (macro): {pr_auc:.4f}")
    return accuracy, recall, f1, pr_auc

# Load your dataset
#data = EllipticDataset(root=processed_data_path)

data = EllipticDataset(root=processed_data_path)
data = data[1]
data = data.to(device)
epochs = 30

# Get all unique time steps
time_steps = data.time_step.unique()

# Dictionary to store each subgraph
subgraphs = {}






test_loader = NeighborLoader(
    data,
    shuffle=True,
    num_neighbors=[10, 10, 25],
    batch_size=32,
    input_nodes= data.test_mask
)


#batch_size = 64

#train_loader = DataLoader(data[data.train_mask], shuffle=True, batch_size=batch_size)
#val_loader = DataLoader(data[data.val_mask], shuffle=True, batch_size=batch_size)
#test_loader = DataLoader(data[data.test_mask], shuffle=False, batch_size=batch_size)

# Define model, optimizer, and loss function
model = GAT(data.num_features, 256, 2,
            8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

with open("training_log_gat_synthetic.txt", "w") as file:
    for epoch in range(epochs):
        epoch_loss = 0.0

        for i, t in enumerate(sorted(time_steps.tolist())):
            # 1. Find node indices for current time step
            node_mask = data.time_step == t
            node_indices = node_mask.nonzero(as_tuple=False).view(-1)

            # 2. Create the subgraph (with relabeling)
            edge_index_t, edge_attr_t = subgraph(
                node_indices,
                data.edge_index,
                edge_attr=getattr(data, 'edge_attr', None),
                relabel_nodes=True
            )

            # 3. Extract node features and labels for the subgraph
            x_t = data.x[node_indices]
            y_t = data.y[node_indices]
            time_step_t = data.time_step[node_indices]

            class_0_idx = (y_t == 0).nonzero(as_tuple=True)[0]
            class_1_idx = (y_t == 1).nonzero(as_tuple=True)[0]

            min_len = min(len(class_0_idx), len(class_1_idx))
            if min_len == 0:
                continue  # Skip this time step if one class is missing

            # Shuffle and sample balanced sets
            perm_0 = class_0_idx[torch.randperm(len(class_0_idx))[:min_len]]
            perm_1 = class_1_idx[torch.randperm(len(class_1_idx))[:min_len]]
            balanced_idx = torch.cat([perm_0, perm_1], dim=0)

            # Shuffle total balanced indices
            balanced_idx = balanced_idx[torch.randperm(len(balanced_idx))]

            # Split into train and validation
            split_idx = int(0.8 * len(balanced_idx))
            train_idx = balanced_idx[:split_idx]
            val_idx = balanced_idx[split_idx:]

            # Create masks
            train_mask_t = torch.zeros(y_t.size(0), dtype=torch.bool)
            val_mask_t = torch.zeros(y_t.size(0), dtype=torch.bool)
            train_mask_t[train_idx] = True
            val_mask_t[val_idx] = True

            # 5. Create subgraph Data object
            data_t = Data(
                x=x_t,
                edge_index=edge_index_t,
                edge_attr=edge_attr_t,
                y=y_t,
                time_step=time_step_t,
                train_mask=train_mask_t,
                val_mask=val_mask_t
            )

            train_loader = NeighborLoader(
                data_t,
                shuffle=True,
                num_neighbors=[-1] * 2,
                batch_size=32,


            )

            val_loader = NeighborLoader(
                data_t,
                shuffle=True,
                num_neighbors=[-1] * 2,
                batch_size=32,

            )

            loss = train(train_loader, model, optimizer, device, criterion)
            epoch_loss += loss

            #log = f"Epoch {epoch + 1:02d}, Time Step {i:02d}, Loss: {loss:.6f}\n"
            #print(log)
            #file.write(log)

        # Validate once per epoch
        validate(val_loader)


if not os.path.exists(os.path.join(trained_model_path, 'modeling_gat_trained.pth')):
    #Run training and validation
    with open("gat_training_log_losses_per_epoch.txt", "w") as file:
        for epoch in range(epochs):
            loss = train(train_loader, model, optimizer, device,
                                         criterion, False)
            log = f"Epoch {epoch+1:02d}, Loss: {loss:.6f}\n"
            print(log)
            file.write(log)
            accuracy, precision, recall, f1, auc_pr = validate(val_loader, model, device, False)

            # Logging
            log = (
                f"gat Metrics --- Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                f"F1: {f1:.4f}, AUC-PR: {auc_pr:.4f}\n"
            )
            print(log)
            file.write(log)


    torch.save(model.state_dict(), os.path.join(trained_model_path, 'modeling_gat_trained.pth'))
else:
    model.load_state_dict(torch.load(os.path.join(trained_model_path, 'modeling_gat_trained.pth')))


print("\n----EVALUATION----\n")
with open(f"evaluation_performance_metrics_gat_trained.txt", "w") as f:
    f.write("----EVALUATION----\n")
    accuracy, precision, recall, f1, pr_auc, confusion_matrix_model, pr_auc_curve, fig = evaluate(model, test_loader, device,
                                                                            'GAT', False)
    f.write("----{gat}----\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"pr_auc Score: {pr_auc:.4f}\n")

