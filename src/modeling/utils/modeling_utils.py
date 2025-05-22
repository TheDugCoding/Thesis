import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, average_precision_score, \
    precision_recall_curve, precision_score
from tqdm import tqdm

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

    return total_loss / total_examples


def validate(val_loader, model, device, framework=False):
    """
    :param val_loader: val_loader of the dataset
    :param model: gnn model to test
    :param device: the device to use
    :param framework: True if the model is the framework
    :return: accuracy, recall, f1, pr_auc
    """
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
    precision = precision_score(true_labels, preds, average='binary', pos_label=0)
    recall = recall_score(true_labels, preds, average='binary', pos_label=0)
    f1 = f1_score(true_labels, preds, average='binary', pos_label=0)

    # PR-AUC
    probs_class0 = probs[:, 0]
    pr_auc = average_precision_score(true_labels, probs_class0, pos_label=0, average='weighted')

    return accuracy, precision, recall, f1, pr_auc


def evaluate(model, test_loader, device, name, framework=False):
    model.eval()
    preds = []
    true = []
    probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
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
    precision = precision_score(true_labels, preds, average='binary', pos_label=0)
    recall = recall_score(true_labels, preds, average='binary', pos_label=0)
    f1 = f1_score(true_labels, preds, average='binary', pos_label=0)

    # PR-AUC
    probs_class0 = probs[:, 0]
    pr_auc = average_precision_score(true_labels, probs_class0, pos_label=0, average='weighted')
    precision_plot, recall_vals, pr_thresholds = precision_recall_curve(true_labels, probs_class0, pos_label=0)

    true_labels = true_labels.numpy()

    confusion_matrix_model = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_model)
    disp.plot()
    plt.title(f'Confusion Matrix {name}')
    plt.savefig(f'confusion_matrix_{name}_plot_.png')
    print(confusion_matrix_model)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_plot, label=f'PR-AUC = {pr_auc:.4f}', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.savefig(f'precision_recall_curve_{name}_plot_.png')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return accuracy, precision, recall, f1, pr_auc, confusion_matrix_model, (precision, recall_vals, pr_thresholds)
