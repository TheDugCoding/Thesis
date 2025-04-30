import os

import matplotlib.pyplot as plt
import torch
import torch_geometric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

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
data = data[1]
epochs = 10

train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data.train_mask
)

val_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data.val_mask
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes= data.test_mask
)

# Define model, optimizer, and loss function
model = GraphSAGE(
    in_channels=data.num_features,
    hidden_channels=256,
    num_layers=2,
    out_channels=2,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
class_counts_int = torch.bincount(data.y).int().tolist()
#w0 = 0.95
#w1= 0.05
#w2= 1/class_counts_int[2]
#weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
#criterion = FocalLoss(alpha=0.25, gamma=2.0)
criterion = torch.nn.CrossEntropyLoss(ignore_index=2)

# Training loop
def train(train_loader):
    model.train()
    total_loss = 0
    total_examples = 0

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index)

        # Only calculate loss for the target (input) nodes, not the neighbors
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.batch_size
        total_examples += batch.batch_size

    return  total_loss / total_examples


#Run training
with open("training_log_graphsage.txt", "w") as file:
    for epoch in range(epochs):
        loss = train(train_loader)
        log = f"Epoch {epoch+1:02d}, Loss: {loss:.6f}\n"
        print(log)
        file.write(log)

        #validation
        model.eval()
        preds = []
        true = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                preds.append(out[:batch.batch_size].argmax(dim=1).cpu())
                true.append(batch.y[:batch.batch_size].cpu())
        preds = torch.cat(preds)
        true_labels = torch.cat(true)
        accuracy = (preds == true_labels).sum().item() / true_labels.size(0)
        recall = recall_score(true_labels, preds, average='macro')
        f1 = f1_score(true_labels, preds, average='macro')

        print(f"Accuracy: {accuracy:.4f}, Recall (macro): {recall:.4f}, F1 Score (macro): {f1:.4f}")

torch.save(model.state_dict(), os.path.join(trained_model_path, 'modeling_graphsage_trained.pth'))

print("\n----EVALUATION----\n")
# Inference
model.eval()
preds = []
true = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        preds.append(out[:batch.batch_size].argmax(dim=1).cpu())
        true.append(batch.y[:batch.batch_size].cpu())

preds = torch.cat(preds)
true_labels = torch.cat(true)
accuracy = (preds == true_labels).sum().item() / true_labels.size(0)
recall = recall_score(true_labels, preds, average='macro')
f1 = f1_score(true_labels, preds, average='macro')

print(f"Final Accuracy: {accuracy:.4f}\n")
print(f"Recall (macro): {recall:.4f}\n")
print(f"F1 Score (macro): {f1:.4f}\n")
print('Confusion matrix')

with open("final_accuracy_graphsage_trained.txt", "w") as f:
    f.write(f"Final Accuracy: {accuracy:.4f}\n")
    f.write(f"Recall (macro): {recall:.4f}\n")
    f.write(f"F1 Score (macro): {f1:.4f}\n")

true_labels = true_labels.numpy()
predicted_labels = preds.numpy()

cm = confusion_matrix(true_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_plot.png')
print(cm)
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_plot.png')  # Save the plot as a PNG file

plt.show()