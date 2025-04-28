import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,f1_score, recall_score
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from src.data_preprocessing.preprocess import EllipticDataset, AmlSimDataset
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/downstream_task/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)


# Load your dataset
data = EllipticDataset(root=processed_data_path)

data = data[0]

"""-----Check missing values-----"""
print("-----Check missing values-----")

def check_data_integrity(data):
    for key, value in data:
        if torch.is_tensor(value):
            num_nans = torch.isnan(value).sum().item()
            num_infs = torch.isinf(value).sum().item()
            print(f"{key}: {num_nans} NaNs, {num_infs} Infs, Shape: {value.shape}")

# Usage
check_data_integrity(data)

"""-----Check class distribution-----"""
print("-----Check class distribution-----")
labels = data.y

# Count the number of occurrences for each class
unique_classes, counts = torch.unique(labels, return_counts=True)

# Total number of samples
total_samples = labels.size(0)

# Print the counts and percentages
for cls, count in zip(unique_classes, counts):
    percentage = (count.item() / total_samples) * 100
    print(f"Class {cls.item()}: {count.item()} samples ({percentage:.2f}%)")