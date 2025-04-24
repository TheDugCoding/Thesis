import os

import matplotlib.pyplot as plt
import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,f1_score, recall_score

from src.data_preprocessing.preprocess import EllipticDataset, RealDataTraining, EllipticDatasetWithoutFeatures
from src.modeling.downstream_task.GAT_elliptic import GAT
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax import DeepGraphInfomax, Encoder
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/final_framework/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)

dataset = RealDataTraining(root=processed_data_path, add_topological_features=False)

EPS = 1e-15

def corruption(x, edge_index, batch_size):
    return x[torch.randperm(x.size(0))], edge_index, batch_size


class DGIPlusGNN(torch.nn.Module):
    def __init__(self, dgi, classifier):
        super().__init__()
        self.dgi = dgi
        self.classifier = classifier

    def forward(self, x, topological_feature, edge_index):
        topological_latent_representation = self.dgi(topological_feature, edge_index)
        x = torch.cat([x, topological_latent_representation], dim=1)
        x = self.classifier(x,edge_index)

        return x


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

# Load your dataset
data = EllipticDataset(root=processed_data_path, add_topological_features=True)
topological_data = EllipticDatasetWithoutFeatures(root=processed_data_path, add_topological_features=True)

data = data[0]

train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data.train_mask
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=32,
    input_nodes=data.test_mask
)

# define the framework, first DGI and then the GNN used in the downstream task
dgi_model = DeepGraphInfomax(
    hidden_channels=64, encoder=Encoder(64, 64, 1),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
# load the pretrained parameters
dgi_model.load_state_dict(torch.load(os.path.join(relative_path_trained_dgi, 'final_framework_trained.pth')))
# freeze layers 2 and 3, let layer 1 learn
dgi_model.conv2.weight.requires_grad = False
dgi_model.conv2.bias.requires_grad = False
dgi_model.conv3.weight.requires_grad = False
dgi_model.conv3.bias.requires_grad = False

#define the downstream GNN
gnn_model = GAT(data.num_features, 64, 3,
            8).to(device)
model = DGIPlusGNN(dgi_model, gnn_model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train(train_loader):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        #take only the topological feature from the dataset
        batch_topological_feature = topological_data.x[batch.n_id].to(device)
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch_topological_feature, batch.edge_index)

        # Only calculate loss for the target (input) nodes, not the neighbors
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    return total_loss / len(train_loader.dataset)

torch.save(model.state_dict(), os.path.join(trained_model_path, 'final_framework_trained.pth'))

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

# plt.show()

