import os

import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
        x = torch.sigmoid(x)
        return x


# Load your dataset
data = EllipticDataset(root = processed_data_path, add_topological_features = True)


# Define model, optimizer, and loss function
model = GAT(data.num_features, 64, 1,
            8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out.view(-1)[data.train_mask], data.y[data.train_mask].float())
    loss.backward()
    optimizer.step()
    return loss.item()


torch.save(model.state_dict(), os.path.join(trained_model_path, 'modeling_gat_trained.pth'))

# Run training
for epoch in range(100):
    loss = train()
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# Inference
model.eval()
preds = model(data.x, data.edge_index).argmax(dim=1)
accuracy = (preds[data.test_mask] == data.y[data.test_mask]).sum().item() / data.y[data.test_mask].size(0)
print(f"Final Accuracy: {accuracy:.4f}")
print('Confusion matrix')
confusion_matrix = confusion_matrix([label.bool().item() for label in data.y[data.test_mask]], [pred.bool().item() for pred in preds[data.test_mask]])
print(confusion_matrix)
ConfusionMatrixDisplay(confusion_matrix).plot()
plt.show()

