import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader

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


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# Load your dataset
data = EllipticDataset(root=processed_data_path, add_topological_features=True)

data = data[0]

train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=64,
    input_nodes=data.train_mask
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

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index)

        # Only calculate loss for the target (input) nodes, not the neighbors
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)




# Run training
with open("training_log_gat_elliptic.txt", "w") as file:
    for epoch in range(30):
        loss = train(train_loader)
        log = f"Epoch {epoch:02d}, Loss: {loss:.6f}\n"
        print(log)
        file.write(log)

torch.save(model.state_dict(), os.path.join(trained_model_path, 'modeling_gat_trained.pth'))

# Inference
model.eval()
with torch.no_grad():
    data = data.to(device)
    preds = model(data.x, data.edge_index).argmax(dim=1)
    accuracy = (preds[data.test_mask] == data.y[data.test_mask]).sum().item() / data.y[data.test_mask].size(0)
print(f"Final Accuracy: {accuracy:.4f}")
print('Confusion matrix')
true_labels = data.y[data.test_mask].cpu().numpy()
predicted_labels = preds[data.test_mask].cpu().numpy()

cm = confusion_matrix(true_labels, predicted_labels)

print(cm)
ConfusionMatrixDisplay(cm).plot()
# Display and save confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix)
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_plot.png')  # Save the plot as a PNG file

# plt.show()
