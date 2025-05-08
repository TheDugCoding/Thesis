from torch_geometric.nn import GraphSAGE

from src.utils import get_data_folder, get_data_sub_folder, get_src_sub_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
relative_path_trained_model = 'modeling/downstream_task/trained_models'
processed_data_path = get_data_sub_folder(relative_path_processed)
trained_model_path = get_src_sub_folder(relative_path_trained_model)




# Define model, optimizer, and loss function
model = GraphSAGE(
    in_channels=65,
    hidden_channels=256,
    num_layers=3,
    out_channels=2,
    aggr='banananananana'
)