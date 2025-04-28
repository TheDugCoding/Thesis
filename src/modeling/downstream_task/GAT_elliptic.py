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

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        #Dropout helps prevent overfitting by randomly nullifying outputs from neurons during the training process. This encourages the network to learn redundant representations for everything and hence, increases the model's ability to generalize.
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# Load your dataset
data = EllipticDataset(root=processed_data_path)

data_train = data[1]

train_loader = NeighborLoader(
    data_train,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data_train.train_mask
)

test_loader = NeighborLoader(
    data_train,
    num_neighbors=[10, 10],
    batch_size=32,  # Adjust depending on memory
    #input_nodes=data_test.test_mask
)

# Define model, optimizer, and loss function
model = GAT(data.num_features, 64, 3,
            8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


# Training loop
def train(train_loader):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index)

        # Only calculate loss for the target (input) nodes, not the neighbors
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    return total_loss / len(train_loader.dataset)


#Run training
with open("training_log_gat_synthetic.txt", "w") as file:
    for epoch in range(100):
        loss = train(train_loader)
        log = f"Epoch {epoch:02d}, Loss: {loss:.6f}\n"
        print(log)
        file.write(log)

torch.save(model.state_dict(), os.path.join(trained_model_path, 'modeling_gat_trained.pth'))

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
print(f"Final Accuracy: {accuracy:.4f}")
print('Confusion matrix')
recall = recall_score(true_labels, preds, average='macro')
f1 = f1_score(true_labels, preds, average='macro')

with open("final_accuracy_gat_trained.txt", "w") as f:
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

