import torch.nn as nn

from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/testing/trained_models'
relative_path_trained_dgi = 'modeling/pre_training/topological_pre_training/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)
trained_dgi_model_path = get_src_sub_folder(relative_path_trained_dgi)

class GraphsageWithMLP(nn.Module):
    def __init__(self, graphsage_model, mlp):
        super().__init__()
        self.graphsage = graphsage_model
        self.mlp = mlp

    def forward(self, x, edge_index):
        x = self.graphsage(x, edge_index)
        return self.mlp(x)