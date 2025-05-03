import os

import matplotlib.pyplot as plt
import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,f1_score, recall_score, average_precision_score

from src.data_preprocessing.preprocess import EllipticDataset
from torch_geometric.nn import GraphSAGE
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax import DeepGraphInfomax, Encoder
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/final_framework/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)

# Training loop
def train(train_loader, model, optimizer, device, criterion, framework=False):
    """
    :param train_loader:
    :param model:
    :param optimizer:
    :param device:
    :param criterion:
    :param framework: if the model is part of the framework the forward pass is different
    :return:
    """
    model.train()
    total_loss = 0
    total_examples = 0

    for batch in tqdm(train_loader, desc="training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        if framework:
            out = model(batch)
        else:
            out = model(batch.x, batch.edge_index)

        # Only calculate loss for the target (input) nodes, not the neighbors
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.batch_size
        total_examples += batch.batch_size

    return  total_loss / total_examples

def validate(val_loader, model, device, framework=False):
    model.eval()
    preds = []
    true = []
    probs = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # Forward pass
            if framework:
                out = model(batch)
            else:
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
    probs_class1 = probs[:, 1]
    auc_pr = average_precision_score(true_labels, probs_class1)

    print(f"Accuracy: {accuracy:.4f}, Recall (macro): {recall:.4f}, F1 Score (macro): {f1:.4f}, AUC-PR (macro): {auc_pr:.4f}")
    return accuracy, recall, f1, auc_pr

def evaluate(model, test_loader, device, name, framework=False):
    model.eval()
    preds = []
    true = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            if framework:
                out = model(batch)
            else:
                out = model(batch.x, batch.edge_index)
            preds.append(out[:batch.batch_size].argmax(dim=1).cpu())
            true.append(batch.y[:batch.batch_size].cpu())

    preds = torch.cat(preds)
    true_labels = torch.cat(true)
    accuracy = (preds == true_labels).sum().item() / true_labels.size(0)
    recall = recall_score(true_labels, preds, average='macro')
    f1 = f1_score(true_labels, preds, average='weighted')

    print(f"Final Accuracy: {accuracy:.4f}\n")
    print(f"Recall (macro): {recall:.4f}\n")
    print(f"F1 Score: {f1:.4f}\n")
    print('Confusion matrix')

    with open(f"performance_metrics_{name}_trained.txt", "w") as f:
        f.write(f"Final Accuracy: {accuracy:.4f}\n")
        f.write(f"Recall (macro): {recall:.4f}\n")
        f.write(f"F1 Score (macro): {f1:.4f}\n")

    true_labels = true_labels.numpy()
    predicted_labels = preds.numpy()

    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_{name}_plot_.png')
    print(cm)

    plt.show()