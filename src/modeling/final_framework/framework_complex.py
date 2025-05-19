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
from src.modeling.pre_training.topological_pre_training.deep_graph_infomax import DeepGraphInfomaxFlexFronts, EncoderFlexFronts, corruptionflexfronts
from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder
from src.modeling.utils.modeling_utils import train, validate

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/final_framework/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)

EPS = 1e-15


class DGIPlusGNN(torch.nn.Module):
    def __init__(self, dgi, downstream_gnn, flipping_layer):
        super().__init__()
        self.dgi = dgi
        self.downstream_gnn = downstream_gnn
        self.flipping_layer = flipping_layer

    def forward(self, batch):
        # it refers to the DGI model and the special "flipping layer"
        x = batch.x
        if self.flipping_layer:
            topological_latent_representation = self.dgi(batch.topological_features, batch.edge_index,
                                              batch.batch_size, framework=True, layer=0)
        else:
            topological_latent_representation = self.dgi(batch.topological_features, batch.edge_index,
                                                         batch.batch_size, framework=True)
        #the DGI encoder returns three values pos_z, neg_z and summary, the latent representation of the graph is pops_z
        x = torch.cat([x, topological_latent_representation[0]], dim=1)
        x = self.downstream_gnn(x, batch.edge_index)

        return x

if __name__ == "__main__":
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
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.train_mask
    )

    val_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.val_mask
    )

    test_loader = NeighborLoader(
        data,
        shuffle=True,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=data.test_mask
    )

    # define the framework, first DGI and then the GNN used in the downstream task
    dgi_model = DeepGraphInfomaxFlexFronts(
        hidden_channels=64, encoder=EncoderFlexFronts(64, 64, 2),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruptionflexfronts).to(device)
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

    # same model as in garphsage_elliptic
    gnn_model_downstream = GraphSAGE(
        in_channels=data.num_features + 64,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    ).to(device)

    model_framework = DGIPlusGNN(dgi_model, gnn_model_downstream, True).to(device)
    optimizer = torch.optim.Adam(model_framework.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)



    if not os.path.exists(os.path.join(trained_model_path, 'final_framework_trained.pth')):
        #Run training
        with open("training_log_framework_losses_per_epoch.txt", "w") as file:
            for epoch in range(5):
                loss = train(train_loader, model_framework, optimizer, device, criterion)
                log = f"Epoch {epoch + 1:02d}, Loss: {loss:.6f}\n"
                print(log)
                file.write(log)
                validate(val_loader, model_framework, device)

        torch.save(model_framework.state_dict(), os.path.join(trained_model_path, 'final_framework_trained.pth'))
    else:
        model_framework.load_state_dict(torch.load(os.path.join(trained_model_path, 'final_framework_trained.pth'), map_location=device))

    print("\n----EVALUATION----\n")
    # Inference
    model_framework.eval()
    preds = []
    true = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            out = model_framework(batch)
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

    with open("performance_metrics_framework_trained.txt", "w") as f:
        f.write(f"Final Accuracy: {accuracy:.4f}\n")
        f.write(f"Recall (macro): {recall:.4f}\n")
        f.write(f"F1 Score (macro): {f1:.4f}\n")

    true_labels = true_labels.numpy()
    predicted_labels = preds.numpy()

    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_plot_framework.png')
    print(cm)

    plt.show()

