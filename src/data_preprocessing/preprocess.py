import networkx as nx
import pandas as pd
import torch
import os
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx
from src.data_preprocessing.utils import get_structural_info
from src.utils import get_data_sub_folder, get_data_folder

script_dir = get_data_folder()
relative_path_processed  = 'processed'
processed_data_location = get_data_sub_folder(relative_path_processed)

#dataset locations
relative_path_aml_world_raw = 'raw/aml_world/small_LI/formatted_transactions.csv'
relative_path_rabobank_raw = 'raw/rabobank/rabobank_data.csv'
relative_path_saml_d_raw = 'raw/saml-d/SAML-D.csv'
relative_path_elliptic_raw = 'raw/elliptic++_dataset/AddrAddr_edgelist.csv'


'''---aml_world dataset preprocessing---'''

def pre_process_aml_world():
    # Check if AMl world has already been preprocessed
    if not os.path.exists(os.path.join(processed_data_location, 'aml_world.graphml')):
        # Load the dataset
        df_aml_world = pd.read_csv(os.path.join(script_dir, relative_path_aml_world_raw))

        # Initialize a directed graph
        G_aml_world = nx.DiGraph()

        # Add edges to the graph from the dataset
        for index, row in df_aml_world.iterrows():
            G_aml_world.add_edge(row['from_id'], row['to_id'],
                       edge_id=row['EdgeID'],
                       timestamp=row['Timestamp'],
                       amount_sent=row['Amount Sent'],
                       sent_currency=row['Sent Currency'],
                       amount_received=row['Amount Received'],
                       received_currency=row['Received Currency'],
                       payment_format=row['Payment Format'],
                       is_laundering=row['Is Laundering'])

            break

            

        G_aml_world = get_structural_info(G_aml_world)

        # Save dataset with additional information
        nx.write_graphml(G_aml_world, os.path.join(processed_data_location, 'aml_world.graphml'))
        return G_aml_world
    else:
        G_aml_world = nx.read_graphml(os.path.join(processed_data_location, 'aml_world.graphml'))

    return G_aml_world

'''---Rabobank dataset preprocessing---'''

def pre_process_rabobank():
    # Check if the Rabobank graph has already been preprocessed
    if not os.path.exists(os.path.join(processed_data_location, 'rabobank.graphml')):
        # Load the dataset
        df_rabobank = pd.read_csv(os.path.join(script_dir, relative_path_rabobank_raw), delimiter=';')

        # Initialize a directed graph
        G_rabobank = nx.DiGraph()

        # Add edges to the graph from the dataset
        for index, row in df_rabobank.iterrows():
            G_rabobank.add_edge(row['start_id'], row['end_id'],
                                total=row['total'],
                                count=row['count'],
                                year_from=row['year_from'],
                                year_to=row['year_to'])

            break

        # Compute additional structural information
        G_rabobank = get_structural_info(G_rabobank)

        # Save dataset with additional information
        nx.write_graphml(G_rabobank, os.path.join(processed_data_location, 'rabobank.graphml'))
    else:
        G_rabobank = nx.read_graphml(os.path.join(processed_data_location, 'rabobank.graphml'))

    return G_rabobank


'''---Enhanced dataset preprocessing---'''

def pre_process_saml_d():
    # Check if the SAML_d graph has already been preprocessed
    if not os.path.exists(os.path.join(processed_data_location, 'saml_d.graphml')):
        # Load the dataset
        df_saml_d = pd.read_csv(os.path.join(script_dir, relative_path_saml_d_raw))

        # Initialize a directed graph
        G_saml_d = nx.DiGraph()

        # Add edges to the graph from the dataset
        for index, row in df_saml_d.iterrows():
            G_saml_d.add_edge(row['Sender_account'], row['Receiver_account'],
                              amount=row['Amount'],
                              payment_currency=row['Payment_currency'],
                              received_currency=row['Received_currency'],
                              sender_bank_location=row['Sender_bank_location'],
                              receiver_bank_location=row['Receiver_bank_location'],
                              payment_type=row['Payment_type'],
                              is_laundering=row['Is_laundering'],
                              laundering_type=row['Laundering_type'])

            break

        # Compute additional structural information
        G_saml_d = get_structural_info(G_saml_d)

        # Save dataset with additional information
        nx.write_graphml(G_saml_d, os.path.join(processed_data_location, 'saml_d.graphml'))
    else:
        G_saml_d = nx.read_graphml(os.path.join(processed_data_location, 'saml_d.graphml'))

    return G_saml_d

'''---Elliptic++ dataset preprocessing---'''

def pre_process_elliptic():
    # Check if the AddrAddr graph has already been preprocessed
    if not os.path.exists(os.path.join(processed_data_location, 'elliptic_addr_addr.graphml')):
        # Load the dataset
        df_addr_addr = pd.read_csv(os.path.join(script_dir, relative_path_elliptic_raw))

        # Initialize a directed graph
        G_addr_addr = nx.DiGraph()

        # Add edges to the graph from the dataset
        for index, row in df_addr_addr.iterrows():
            G_addr_addr.add_edge(row['input_address'], row['output_address'])

            break

        # Compute additional structural information
        G_addr_addr = get_structural_info(G_addr_addr)

        # Save dataset with additional information
        nx.write_graphml(G_addr_addr, os.path.join(processed_data_location, 'elliptic_addr_addr.graphml'))

    else:
        G_addr_addr = nx.read_graphml(os.path.join(processed_data_location, 'elliptic_addr_addr.graphml'))

    return G_addr_addr

# Function to convert a NetworkX graph to PyG data
def convert_to_pyg(G):
    G = get_structural_info(G)
    data = from_networkx(G, group_node_attrs=["degree", "degree_centrality", "pagerank"])
    return data

# Function to preprocess a dataset
def preprocess_dataset(preprocess_func):
    G = preprocess_func()
    return convert_to_pyg(G)

# Custom PyG dataset class
class FinancialGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []  # No raw files, since graphs are pre-processed elsewhere.

    @property
    def processed_file_names(self):
        return [
            'financial_dataset_0.pt',
            'financial_dataset_1.pt',
            'financial_dataset_2.pt',
            'financial_dataset_3.pt'
        ]

    def process(self):
        """Processes raw data into PyG data objects and saves them as .pt files."""
        # Generate the graph data from pre-processing functions
        data_list = [
            from_networkx(pre_process_aml_world(), group_node_attrs=["degree", "degree_centrality", "pagerank"]),
            from_networkx(pre_process_rabobank(), group_node_attrs=["degree", "degree_centrality", "pagerank"]),
            from_networkx(pre_process_saml_d(), group_node_attrs=["degree", "degree_centrality", "pagerank"]),
            from_networkx(pre_process_elliptic(), group_node_attrs=["degree", "degree_centrality", "pagerank"]),
        ]

        # Save each graph as a separate .pt file
        for idx, data in enumerate(data_list):
            data = Data(x=data.x, edge_index=data.edge_index)
            torch.save(data, os.path.join(self.processed_dir, f'financial_dataset_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Loads and returns the graph at the given index."""
        return torch.load(os.path.join(self.processed_dir, f'financial_dataset_{idx}.pt'))


# Usage
'''
dataset = FinancialGraphDataset(root = processed_data_location)
print("Dataset saved at:", os.path.join(relative_path_processed, 'processed/financial_graphs.pt'))
print(dataset[0])  # Print first graph
print(dataset[1])  # Print first graph

dataset = FinancialGraphDataset(root="data/financial_graphs")
print(len(dataset))  # Should print 4 (since we have 4 processed graphs)

graph_0 = dataset[0]  # Load the first graph
print(graph_0)  # Print PyG data object

print('done')
print('hi')
print('done')
'''