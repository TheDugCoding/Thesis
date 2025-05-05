import torch
from torch_geometric.nn import GraphSAGE, GAT

from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/testing/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)


#define here the models to test against the framework

def model_list(data):
    """

    :param data: the dataset that is used
    :return: a dict containing all the gnns to test against the framework
    """

    #list of models  to test

    """----GRAPHSAGE----"""
    gnn_model_graphsage = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=3,
        out_channels=2,
    )
    optimizer_gnn_graphsage = torch.optim.Adam(gnn_model_graphsage.parameters(), lr=0.005, weight_decay=5e-4)
    criterion_gnn_graphsage = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """----GAT----"""
    gnn_model_gat = GAT(
        in_channels=data.num_features,
        hidden_channels = 256,
        num_layers = 2,
        out_channels = 2,
        heads = 8)
    optimizer_gnn_gat = torch.optim.Adam(gnn_model_graphsage.parameters(), lr=0.005, weight_decay=5e-4)
    criterion_gnn_gat = torch.nn.CrossEntropyLoss(ignore_index=-1)

    """---------------------------------------------"""
    # Store all in a nested dict
    model_dict = {
        'graphsage': {
            'model': gnn_model_graphsage,
            'optimizer': optimizer_gnn_graphsage,
            'criterion': criterion_gnn_graphsage
        },

        'gat': {
            'model': gnn_model_gat,
            'optimizer': optimizer_gnn_gat,
            'criterion': criterion_gnn_gat
        }
    }


    return model_dict