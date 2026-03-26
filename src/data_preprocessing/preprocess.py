import os
import pickle

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import from_networkx, to_networkx
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


from src.data_preprocessing.utils import get_structural_info
from src.utils import get_data_sub_folder, get_data_folder

script_dir = get_data_folder()
#script_dir = "D:/University/thesis_dataset"
relative_path_processed = 'processed'
processed_data_location = get_data_sub_folder(relative_path_processed)
#processed_data_location = "D:/University/thesis_dataset/processed"

# dataset locations
relative_path_aml_sim_trans = 'raw/aml_sim_banks/transactions.csv'
relative_path_aml_sim_nodes = 'raw/aml_sim_banks/accounts.csv'
relative_path_aml_world_raw = 'raw/aml_world/small_LI/formatted_transactions.csv'
relative_path_rabobank_raw = 'raw/rabobank/rabobank_data.csv'
relative_path_saml_d_raw = 'raw/saml-d/SAML-D.csv'
relative_path_elliptic_raw_edges = 'raw/elliptic++_dataset/AddrAddr_edgelist.csv'
relative_path_elliptic_raw_node_features = 'raw/elliptic++_dataset/wallets_features_classes_combined.csv'
relative_path_ethereum_raw = 'raw/ethereum_phishing/MulDiGraph.pkl'
relative_path_erc_20_raw = 'raw/ERC20-stablecoins/token_transfers.csv'

'''---aml_sim dataset preprocessing---'''


def pre_process_aml_sim():
    # Check if AMl world has already been preprocessed
    if not os.path.exists(os.path.join(processed_data_location, 'aml_sim.graphml')):
        # Load the dataset
        df_aml_sim_trans = pd.read_csv(os.path.join(script_dir, relative_path_aml_sim_trans))
        df_aml_sim_nodes = pd.read_csv(os.path.join(script_dir, relative_path_aml_sim_nodes))

        # Initialize a directed graph
        G_aml_sim = nx.DiGraph()

        # Convert categorical values from transaction.csv into numerical
        tx_type_mapping = {'TRANSFER': 1}  # Add more types as needed
        df_aml_sim_trans['tx_type'] = df_aml_sim_trans['tx_type'].map(tx_type_mapping)
        df_aml_sim_trans['is_sar'] = df_aml_sim_trans['is_sar'].astype(int)
        df_aml_sim_trans['tran_timestamp'] = pd.to_datetime(df_aml_sim_trans['tran_timestamp']).astype(
            'int64') / 10 ** 9

        # Add edges to the graph from the dataset
        for index, row in df_aml_sim_trans.iterrows():
            G_aml_sim.add_edge(row['orig_acct'], row['bene_acct'],
                               tran_id=row['tran_id'],
                               tx_type=row['tx_type'],
                               base_amt=row['base_amt'],
                               tran_timestamp=row['tran_timestamp'],
                               is_sar=row['is_sar'],
                               alert_id=row['alert_id'])

        # Convert categorical values from account.csv into numerical
        df_aml_sim_nodes['prior_sar_count'] = df_aml_sim_nodes['prior_sar_count'].astype(int)
        df_aml_sim_nodes['open_dt'] = pd.to_datetime(df_aml_sim_nodes['open_dt'], errors='coerce').astype(
            'int64') / 10 ** 9
        df_aml_sim_nodes['close_dt'] = pd.to_datetime(df_aml_sim_nodes['close_dt'], errors='coerce').astype(
            'int64') / 10 ** 9
        bank_mapping = {'bank_a': 1, 'bank_b': 2, 'bank_c': 3}  # Define mapping
        df_aml_sim_nodes['bank_id'] = df_aml_sim_nodes['bank_id'].map(bank_mapping).fillna(-1).astype(int)

        # Add node features only if they have values
        for index, row in df_aml_sim_nodes.iterrows():
            if row['acct_id'] in G_aml_sim.nodes:
                G_aml_sim.nodes[row['acct_id']].update({
                    'acct_id': row['acct_id'],
                    'prior_sar_count': row['prior_sar_count'],
                    'open_dt': row['open_dt'],
                    'close_dt': row['close_dt'],
                    'initial_deposit': row['initial_deposit'],
                    'bank_id': row['bank_id'],
                })

        G_aml_sim = get_structural_info(G_aml_sim)

        # Save dataset with additional information
        nx.write_graphml(G_aml_sim, os.path.join(processed_data_location, 'aml_sim.graphml'))
        return G_aml_sim
    else:
        G_aml_sim = nx.read_graphml(os.path.join(processed_data_location, 'aml_sim.graphml'))

    return G_aml_sim


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
            start_id = row['start_id']
            end_id = row['end_id']

            # Add dummy node feature if node not already present
            if start_id not in G_rabobank:
                G_rabobank.add_node(start_id, dummy=0)
            if end_id not in G_rabobank:
                G_rabobank.add_node(end_id, dummy=0)

            # Add edge with attributes
            G_rabobank.add_edge(start_id, end_id,
                                total=row['total'],
                                count=row['count'],
                                year_from=row['year_from'],
                                year_to=row['year_to'])

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

        # Compute additional structural information
        G_saml_d = get_structural_info(G_saml_d)

        # Save dataset with additional information
        nx.write_graphml(G_saml_d, os.path.join(processed_data_location, 'saml_d.graphml'))
    else:
        G_saml_d = nx.read_graphml(os.path.join(processed_data_location, 'saml_d.graphml'))

    return G_saml_d


'''---Elliptic++ dataset preprocessing---'''


def pre_process_elliptic():

    # variable used to check for missing graphs
    graphs = []
    df_edges = None
    df_features = None

    # we use the first 42 timestamps because in later timestamps there are disruptions
    for graph_timestep in range(1, 43):

        graphml_path = os.path.join(processed_data_location, f'elliptic_addr_addr_timestep_{graph_timestep}.graphml')

        if not os.path.exists(graphml_path):

            #check if the csv are loaded
            if df_edges is None:
                df_edges = pd.read_csv(os.path.join(script_dir, relative_path_elliptic_raw_edges)).drop_duplicates()
            if df_features is None:
                df_features = pd.read_csv(os.path.join(script_dir, relative_path_elliptic_raw_node_features))

            # from the elliptic ++ dataset
            # https://github.com/git-disl/EllipticPlusPlus/blob/main/Actors%20Dataset/Elliptic%2B%2B_Actors_ActorInteraction_Graph_Viz.ipynb
            # we create a graph for each timestep
            all_ids = df_features[(df_features['Time step'] == graph_timestep)]['address']
            short_edges = df_edges[df_edges['input_address'].isin(all_ids) & df_edges['output_address'].isin(all_ids)]
            graph = nx.from_pandas_edgelist(short_edges, source='input_address', target='output_address',
                                            create_using=nx.DiGraph())

            # add node features
            nodes_features = df_features[df_features['Time step'] == graph_timestep]
            graph.add_nodes_from((row['address'], row.to_dict()) for _, row in nodes_features.iterrows())

            # Add structural features
            graph = get_structural_info(graph)
            nx.write_graphml(graph, graphml_path)
            graphs.append(graph)
        else:
            graphs.append(nx.read_graphml(graphml_path))

    return graphs


'''--- ethereum dataset preprocessing---'''


def pre_process_ethereum():
    # Check if the AddrAddr graph has already been preprocessed
    if not os.path.exists(os.path.join(processed_data_location, 'ethereum.graphml')):

        with open(os.path.join(script_dir, relative_path_ethereum_raw), 'rb') as f:
            G = pickle.load(f)

        # Add dummy feature to every node
        for node in G.nodes:
            G.nodes[node]['dummy'] = 0

        # Compute additional structural information
        G_ethereum = get_structural_info(G)

        # Save dataset with additional information
        nx.write_graphml(G_ethereum, os.path.join(processed_data_location, 'ethereum.graphml'))

    else:
        G_ethereum = nx.read_graphml(os.path.join(processed_data_location, 'ethereum.graphml'))

    return G_ethereum



"""---ERC20-stablecoin---"""

def pre_process_erc_20_stablecoin():

    graphml_path = os.path.join(processed_data_location, 'erc_20_stablecoin.graphml')
    if not os.path.exists(graphml_path):
        # Load the dataset
        df_erc20 = pd.read_csv(os.path.join(script_dir,relative_path_erc_20_raw))

        # Initialize a directed graph
        G = nx.DiGraph()

        node_count = 0
        edge_count = 0

        # Iterate through the dataset and add edges until 600,000 unique nodes are reached
        for i, (_, row) in enumerate(tqdm(df_erc20.iterrows(), total=len(df_erc20))):

            node_count += 1
            edge_count += 1

            from_addr = row['from_address']
            to_addr = row['to_address']

            G.add_edge(from_addr, to_addr,
                             value=row['value'],
                             time_stamp=row['time_stamp'])



            # Stop if we've reached at least 600000 unique nodes or 1,200,000 edges
            if i + 1 < len(df_erc20):  # Make sure next row exists
                current_timestep = row['time_stamp']
                next_timestep = df_erc20.iloc[i + 1]['time_stamp']

                if (node_count >= 1200000 or edge_count >= 2500000) and current_timestep != next_timestep:
                    print("Stopping condition met.")
                    break

        for node in G.nodes:
            G.nodes[node]['dummy'] = 0

        print("calculating structural information")
        # Compute additional structural information
        G_erc20 = get_structural_info(G)
        
        # Save the graph
        nx.write_graphml(G_erc20, graphml_path)
    else:
        # Load the preprocessed graph
        G_erc20 = nx.read_graphml(graphml_path)

    return G_erc20


# Custom PyG dataset class
class FinancialGraphDatasetOnlyTopologicalFeatures(Dataset):
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


# Custom PyG dataset class, here there is the elliptic dataset
class EllipticDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []  # No raw files, since graphs are pre-processed elsewhere.

    @property
    def processed_file_names(self):
        return [f'ellipticdataset_{i}.pt' for i in range(42)]

    def process(self):
        """Processes raw data into PyG data objects and saves them as .pt files."""
        graphs = pre_process_elliptic()

        # Remove from train/val graphs any nodes that appear in test graphs
        # to prevent data leakage
        test_node_ids = set()
        for graph in graphs[36:42]:
            test_node_ids.update(graph.nodes())

        for i in range(36):
            overlapping = set(graphs[i].nodes()) & test_node_ids
            if overlapping:
                graphs[i].remove_nodes_from(overlapping)

        node_attrs = [
            # we do not include address in the features, the values calculated in pre processing matches the one loaded from the pyg neyworkx
            # structural features
            "degree", "pagerank_normalized", "eigenvector_centrality_norm",
            "clustering_coef",

            # core features
            "class", "num_txs_as_sender", "num_txs_as receiver", "first_block_appeared_in",
            "last_block_appeared_in", "lifetime_in_blocks", "total_txs", "first_sent_block",
            "first_received_block",
            "num_timesteps_appeared_in", "btc_transacted_total", "btc_transacted_min", "btc_transacted_max",
            "btc_transacted_mean", "btc_transacted_median", "btc_sent_total", "btc_sent_min", "btc_sent_max",
            "btc_sent_mean", "btc_sent_median", "btc_received_total", "btc_received_min", "btc_received_max",
            "btc_received_mean", "btc_received_median", "fees_total", "fees_min", "fees_max", "fees_mean",
            "fees_median",
            "fees_as_share_total", "fees_as_share_min", "fees_as_share_max", "fees_as_share_mean",
            "fees_as_share_median",
            "blocks_btwn_txs_total", "blocks_btwn_txs_min", "blocks_btwn_txs_max", "blocks_btwn_txs_mean",
            "blocks_btwn_txs_median", "blocks_btwn_input_txs_total", "blocks_btwn_input_txs_min",
            "blocks_btwn_input_txs_max", "blocks_btwn_input_txs_mean", "blocks_btwn_input_txs_median",
            "blocks_btwn_output_txs_total", "blocks_btwn_output_txs_min", "blocks_btwn_output_txs_max",
            "blocks_btwn_output_txs_mean", "blocks_btwn_output_txs_median", "num_addr_transacted_multiple",
            "transacted_w_address_total", "transacted_w_address_min", "transacted_w_address_max",
            "transacted_w_address_mean", "transacted_w_address_median", "Time step"
        ]

        # convert all graphs in pyg data format
        data_list = []
        for graph in graphs:
            pyg_elliptic = from_networkx(graph, group_node_attrs=node_attrs)

            topological_features = pyg_elliptic.x[:, [0, 1, 2, 3]]
            x = pyg_elliptic.x[:, 5:60]
            y = (pyg_elliptic.x[:, 4] - 1).long()
            time_step = pyg_elliptic.x[:, 60]

            data_list.append(Data(
                x=x,
                edge_index=pyg_elliptic.edge_index,
                topological_features=topological_features,
                y=y,
                time_step=time_step
            ))

        # compute mean and std from graphs 0-35 only to avoid data leakage from test graphs (36-41)
        x_train = torch.cat([d.x for d in data_list[:36]], dim=0)
        topo_train = torch.cat([d.topological_features for d in data_list[:36]], dim=0)

        x_mean, x_std = x_train.mean(dim=0), x_train.std(dim=0)
        topo_mean, topo_std = topo_train.mean(dim=0), topo_train.std(dim=0)

        # avoid division by zero for constant features
        x_std[x_std == 0] = 1
        topo_std[topo_std == 0] = 1

        # standardize all graphs and save
        for i, data in enumerate(data_list):
            data.x = (data.x - x_mean) / x_std
            data.topological_features = (data.topological_features - topo_mean) / topo_std
            # unlabel class 2 (unknown) by replacing with -1
            data.y[data.y == 2] = -1

            labeled_mask = data.y != -1
            if i < 29:
                # balance classes for training graphs; RandomNodeSplit also sets val/test masks to all-False
                min_count = min((data.y == 0).sum().item(), (data.y == 1).sum().item())
                data = RandomNodeSplit(split='random', num_train_per_class=min_count, num_val=0.0, num_test=0.0)(data)
            elif i < 36:
                # validation graphs
                data.val_mask = labeled_mask
            else:
                # test graphs
                data.test_mask = labeled_mask

            torch.save(data, self.processed_paths[i])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Loads and returns the graph at the given index."""
        data = torch.load(os.path.join(self.processed_dir, f'ellipticdataset_{idx}.pt'), weights_only=False)
        return data


# Custom PyG dataset class
class EllipticDatasetWithoutFeatures(Dataset):
    def __init__(self, root, add_topological_features=False, transform=None, pre_transform=None, pre_filter=None):
        self.add_topological_features = add_topological_features
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []  # No raw files, since graphs are pre-processed elsewhere.

    @property
    def processed_file_names(self):
        return ['ellipticdatasetwithoutfeatures.pt']

    def process(self):
        """Processes raw data into PyG data objects and saves them as .pt files."""
        # Generate the graph data from pre-processing functions

        if (self.add_topological_features):
            pyg_elliptic = from_networkx(pre_process_elliptic(), group_node_attrs=[
                # structural features
                "degree", "pagerank_normalized", "eigenvector_centrality_norm",
                "clustering_coef", "class"
            ])

            x = pyg_elliptic.x[:, [
                                      0, 1, 2, 3
                                  ]]
            y = (pyg_elliptic.x[:, 4] - 1).long()

        else:
            pyg_elliptic = from_networkx(pre_process_elliptic(), group_node_attrs=[
                "degree", "class"])

            x = pyg_elliptic.x[:, [0]]
            y = (pyg_elliptic.x[:, 1] - 1).long()

        #pyg_elliptic.x = pyg_elliptic.x.float()

        # Create and save the PyG Data object, in future add the edge features if required
        data = Data(x=x, edge_index=pyg_elliptic.edge_index, y=y)
        node_transform = RandomNodeSplit(split="train_rest",num_val=0.0,num_test=0.2)
        data = node_transform(data)

        torch.save(data, self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Loads and returns the graph at the given index."""
        return torch.load(os.path.join(self.processed_dir, f'ellipticdatasetwithoutfeatures.pt'))

# Custom PyG dataset class
class AmlSimDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []  # No raw files, since graphs are pre-processed elsewhere.

    @property
    def processed_file_names(self):
        return ['aml_sim_dataset.pt']

    def process(self):
        """Processes raw data into PyG data objects and saves them as .pt files."""

        # Generate the graph data
        G_aml_sim = pre_process_aml_sim()

        # Extract available node features dynamically
        first_node = next(iter(G_aml_sim.nodes(data=True)))[1]  # Get the attributes of the first node


        # Convert NetworkX graph to PyG format
        data = from_networkx(G_aml_sim,
                             group_node_attrs=['acct_id', 'prior_sar_count', 'open_dt', 'close_dt', 'initial_deposit',
                                               'bank_id'],
                             group_edge_attrs=['tran_id', 'tx_type', 'base_amt',
                                               'tran_timestamp', 'alert_id'])

        # select all the attributes except 'prior_sar_count', which is the target variable
        x = data.x[:, [0, 2, 3, 4, 5]]
        y = data.x[:, 1].long()

        # Create and save the PyG Data object
        data = Data(x=x, edge_index=data.edge_index, y=y)
        node_transform = RandomNodeSplit(split="train_rest", num_val=0.0, num_test=0.2)
        data = node_transform(data)

        torch.save(data, os.path.join(self.processed_dir, 'aml_sim_dataset.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Loads and returns the graph at the given index."""
        return torch.load(os.path.join(self.processed_dir, f'aml_sim_dataset.pt'))


# Custom PyG dataset class
class RealDataTraining(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []  # No raw files, since graphs are pre-processed elsewhere.

    @property
    def processed_file_names(self):
        return ['real_data_training_dataset_0.pt', 'real_data_training_dataset_1.pt', 'real_data_training_dataset_2.pt']

    def process(self):
        """Processes raw data into PyG data objects and saves them as .pt files."""

        pyg_aml_rabobank = from_networkx(pre_process_rabobank(),
                                         group_node_attrs=[
                                             "dummy", "degree", "pagerank_normalized",
                                             "eigenvector_centrality_norm", "clustering_coef"
                                         ],
                                         group_edge_attrs=["total", "count", "year_from", "year_to"])
        pyg_ethereum = from_networkx(pre_process_ethereum(),
                                     group_node_attrs=[
                                         "dummy", "degree", "pagerank_normalized",
                                         "eigenvector_centrality_norm", "clustering_coef"
                                     ],
                                     group_edge_attrs=["amount", "timestamp"])
        pyg_ecr_20 = from_networkx(pre_process_erc_20_stablecoin(),
                                     group_node_attrs=[
                                         "dummy", "degree", "pagerank_normalized",
                                         "eigenvector_centrality_norm", "clustering_coef"
                                     ],
                                     group_edge_attrs=["value", "time_stamp"])

        # add a new variable for the topological features
        topological_features_rabo = pyg_aml_rabobank.x[:, [1, 2, 3, 4]]
        # store the remaining
        x_rabo = pyg_aml_rabobank.x[:, [0]]

        # Create and save the PyG Data object, in future add the edge features if required
        data_rabo = Data(x=x_rabo, edge_index=pyg_aml_rabobank.edge_index, edge_attr=pyg_aml_rabobank.edge_attr, topological_features=topological_features_rabo)
        node_transform = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.2)
        data_rabo = node_transform(data_rabo)

        # add a new variable for the topological features
        topological_features_ethereum = pyg_ethereum.x[:, [1, 2, 3, 4]]
        # store the remaining
        x_ethereum = pyg_ethereum.x[:, [0]]

        # Create and save the PyG Data object, in future add the edge features if required
        data_ethereum = Data(x=x_ethereum, edge_index=pyg_ethereum.edge_index,
                         edge_attr=pyg_ethereum.edge_attr, topological_features=topological_features_ethereum)
        node_transform = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.2)
        data_ethereum = node_transform(data_ethereum)

        # add a new variable for the topological features
        topological_features_ecr_20 = pyg_ecr_20.x[:, [ 1, 2, 3, 4]]
        # store the remaining
        x_ecr_20 = pyg_ecr_20.x[:, [0]]

        # Create and save the PyG Data object, in future add the edge features if required
        data_ecr_20 = Data(x=x_ecr_20, edge_index=pyg_ecr_20.edge_index,
                             edge_attr=pyg_ecr_20.edge_attr, topological_features=topological_features_ecr_20)
        node_transform = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.2)
        data_ecr_20 = node_transform(data_ecr_20)

        # Save
        torch.save(data_rabo, os.path.join(self.processed_dir, 'real_data_training_dataset_0.pt'))
        torch.save(data_ethereum, os.path.join(self.processed_dir, 'real_data_training_dataset_1.pt'))
        torch.save(data_ecr_20, os.path.join(self.processed_dir, 'real_data_training_dataset_2.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Loads and returns the graph at the given index."""
        data = torch.load(os.path.join(self.processed_dir, f'real_data_training_dataset_{idx}.pt'))
        return data


# Custom PyG dataset class
class AmlTestDataset(Dataset):
    def __init__(self, root, add_topological_features=False, transform=None, pre_transform=None, pre_filter=None):
        self.add_topological_features = add_topological_features
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # No raw files, since graphs are pre-processed elsewhere.

    @property
    def processed_file_names(self):
        return ['aml_test_dataset.pt']

    def process(self):
        """Processes raw data into PyG data objects and saves them as .pt files."""

        if (self.add_topological_features):
            pyg_aml_rabobank = from_networkx(pre_process_aml_world(), group_node_attrs=[
                "degree", "pagerank_normalized", "eigenvector_centrality_norm", "clustering_coef"
            ],
            group_edge_attrs = ["timestamp","amount_sent","sent_currency","amount_received",
                "received_currency","payment_format"]
            )
        else:
            pyg_aml_rabobank = from_networkx(pre_process_aml_world(), group_node_attrs=[
                "degree"], group_edge_attrs = ["timestamp","amount_sent","sent_currency","amount_received",
                "received_currency","payment_format"])

        pyg_aml_rabobank.x = pyg_aml_rabobank.x.float()

        torch.save(pyg_aml_rabobank, self.processed_paths[0])

    def len(self):
        return len(self.data.x)

    def get(self, idx):
        """Loads and returns the graph at the given index."""
        return self.data

def analyze_elliptic_dataset(dataset, save_path=None, check_connectivity=False):
    """
    Prints per-graph statistics for every timestep in the EllipticDataset.

    Covers per graph:
      - Node / edge counts
      - Class distribution (illicit / licit / unknown) and imbalance ratio
      - NaN counts in x and topological_features
      - Mask sizes (train / val / test)
      - Per-feature descriptive statistics (mean, std, min, max, quartiles)
      - Connected components (optional, slow on large graphs)

    Also prints a compact cross-timestep summary table and optionally saves
    everything to a .txt file + a _summary.csv file.
    """
    import pandas as pd

    feature_names = [
        "num_txs_as_sender", "num_txs_as_receiver", "first_block_appeared_in",
        "last_block_appeared_in", "lifetime_in_blocks", "total_txs", "first_sent_block",
        "first_received_block", "num_timesteps_appeared_in",
        "btc_transacted_total", "btc_transacted_min", "btc_transacted_max",
        "btc_transacted_mean", "btc_transacted_median",
        "btc_sent_total", "btc_sent_min", "btc_sent_max", "btc_sent_mean", "btc_sent_median",
        "btc_received_total", "btc_received_min", "btc_received_max",
        "btc_received_mean", "btc_received_median",
        "fees_total", "fees_min", "fees_max", "fees_mean", "fees_median",
        "fees_as_share_total", "fees_as_share_min", "fees_as_share_max",
        "fees_as_share_mean", "fees_as_share_median",
        "blocks_btwn_txs_total", "blocks_btwn_txs_min", "blocks_btwn_txs_max",
        "blocks_btwn_txs_mean", "blocks_btwn_txs_median",
        "blocks_btwn_input_txs_total", "blocks_btwn_input_txs_min",
        "blocks_btwn_input_txs_max", "blocks_btwn_input_txs_mean",
        "blocks_btwn_input_txs_median",
        "blocks_btwn_output_txs_total", "blocks_btwn_output_txs_min",
        "blocks_btwn_output_txs_max", "blocks_btwn_output_txs_mean",
        "blocks_btwn_output_txs_median",
        "num_addr_transacted_multiple",
        "transacted_w_address_total", "transacted_w_address_min",
        "transacted_w_address_max", "transacted_w_address_mean",
        "transacted_w_address_median",
    ]
    topo_names = ["degree", "pagerank_normalized", "eigenvector_centrality_norm", "clustering_coef"]

    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    summary_rows = []

    for i in range(len(dataset)):
        g = dataset[i]
        timestep = i + 1

        n_nodes = g.num_nodes
        n_edges = g.num_edges

        # ── class distribution ───────────────────────────────────────────────
        n_illicit = (g.y == 0).sum().item()
        n_licit   = (g.y == 1).sum().item()
        n_unknown = (g.y == -1).sum().item()
        labeled   = n_illicit + n_licit
        illicit_pct     = 100 * n_illicit / labeled if labeled > 0 else float('nan')
        licit_pct       = 100 * n_licit   / labeled if labeled > 0 else float('nan')
        imbalance_ratio = n_licit / n_illicit if n_illicit > 0 else float('inf')

        # ── NaN counts ───────────────────────────────────────────────────────
        x_nan    = torch.isnan(g.x).sum().item()
        topo_nan = torch.isnan(g.topological_features).sum().item()

        # ── mask sizes ───────────────────────────────────────────────────────
        n_train = g.train_mask.sum().item() if hasattr(g, 'train_mask') and g.train_mask is not None else 0
        n_val   = g.val_mask.sum().item()   if hasattr(g, 'val_mask')   and g.val_mask   is not None else 0
        n_test  = g.test_mask.sum().item()  if hasattr(g, 'test_mask')  and g.test_mask  is not None else 0

        # ── connectivity (optional) ──────────────────────────────────────────
        if check_connectivity:
            G_nx = to_networkx(g, to_undirected=True)
            n_components = nx.number_connected_components(G_nx)
            isolated = sum(1 for n in G_nx.nodes() if G_nx.degree(n) == 0)
        else:
            n_components = None
            isolated     = None

        # ── feature dataframes ───────────────────────────────────────────────
        x_df    = pd.DataFrame(g.x.numpy(),                    columns=feature_names)
        topo_df = pd.DataFrame(g.topological_features.numpy(), columns=topo_names)

        # ── print block ──────────────────────────────────────────────────────
        sep = "=" * 72
        log()
        log(sep)
        log(f"  TIMESTEP {timestep:02d}")
        log(sep)
        log(f"  Nodes              : {n_nodes}")
        log(f"  Edges              : {n_edges}")
        log(f"  Avg degree         : {2 * n_edges / n_nodes:.2f}" if n_nodes > 0 else "  Avg degree : N/A")
        if check_connectivity:
            log(f"  Connected comps    : {n_components}")
            log(f"  Isolated nodes     : {isolated}")
        log()
        log(f"  -- Class distribution --")
        log(f"  Illicit  (0)       : {n_illicit:>6}  ({illicit_pct:.1f}% of labeled)")
        log(f"  Licit    (1)       : {n_licit:>6}  ({licit_pct:.1f}% of labeled)")
        log(f"  Unknown  (-1)      : {n_unknown:>6}")
        log(f"  Labeled total      : {labeled:>6}")
        log(f"  Imbalance ratio    : 1 illicit per {imbalance_ratio:.2f} licit")
        log()
        log(f"  -- NaN counts --")
        log(f"  x NaNs             : {x_nan} / {g.x.numel()}")
        log(f"  topo NaNs          : {topo_nan} / {g.topological_features.numel()}")
        log()
        log(f"  -- Mask sizes --")
        log(f"  Train              : {n_train}")
        log(f"  Val                : {n_val}")
        log(f"  Test               : {n_test}")
        log()
        log(f"  -- Feature statistics (x, {g.x.shape[1]} features) --")
        log(x_df.describe().round(4).to_string())
        log()
        log(f"  -- Topological feature statistics --")
        log(topo_df.describe().round(4).to_string())

        summary_rows.append({
            'timestep':        timestep,
            'n_nodes':         n_nodes,
            'n_edges':         n_edges,
            'n_illicit':       n_illicit,
            'n_licit':         n_licit,
            'n_unknown':       n_unknown,
            'illicit_pct':     round(illicit_pct, 2),
            'imbalance_ratio': round(imbalance_ratio, 2),
            'x_nans':          x_nan,
            'topo_nans':       topo_nan,
            'train_nodes':     n_train,
            'val_nodes':       n_val,
            'test_nodes':      n_test,
            **({"n_components": n_components, "isolated_nodes": isolated} if check_connectivity else {}),
        })

    # ── cross-timestep summary ───────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    log()
    log("=" * 72)
    log("  CROSS-TIMESTEP SUMMARY")
    log("=" * 72)
    log(summary_df.to_string(index=False))

    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines))
        csv_path = save_path.replace('.txt', '_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSaved report  → {save_path}")
        print(f"Saved summary → {csv_path}")

    return summary_df


def visualize_elliptic_dataset(dataset, save_path=None):
    """
    Two-panel visualization for the EllipticDataset:
      1. Class distribution (illicit / licit / unknown) across all 42 timesteps.
      2. Graph structure for a single timestep, nodes coloured by class.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ── 1. class distribution across timesteps ──────────────────────────────
    illicit, licit, unknown = [], [], []
    for i in range(len(dataset)):
        g = dataset[i]
        illicit.append((g.y == 0).sum().item())
        licit.append((g.y == 1).sum().item())
        unknown.append((g.y == -1).sum().item())

    timesteps = list(range(1, 43))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ax = axes[0]
    ax.bar(timesteps, illicit, label='Illicit (0)', color='crimson', alpha=0.8)
    ax.bar(timesteps, licit, bottom=illicit, label='Licit (1)', color='steelblue', alpha=0.8)
    ax.bar(timesteps, unknown, bottom=[i + l for i, l in zip(illicit, licit)],
           label='Unknown (-1)', color='lightgrey', alpha=0.8)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Number of nodes')
    ax.set_title('Class distribution per timestep')
    ax.legend()

    # ── 2. graph structure for timestep 1 (index 0) ─────────────────────────
    g = dataset[0]
    G = to_networkx(g, to_undirected=True)

    # sample a subgraph of max 300 nodes so the layout is readable
    max_nodes = 40000
    if G.number_of_nodes() > max_nodes:
        sampled_nodes = list(G.nodes())[:max_nodes]
        G = G.subgraph(sampled_nodes).copy()

    color_map = {0: 'crimson', 1: 'steelblue', -1: 'lightgrey'}
    node_colors = [color_map.get(g.y[n].item(), 'lightgrey') for n in G.nodes()]

    ax2 = axes[1]
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, ax=ax2, node_color=node_colors,
                     node_size=30, with_labels=False, edge_color='#cccccc', width=0.5)
    ax2.set_title('Graph structure – Timestep 1 (first 300 nodes)')
    legend_handles = [
        mpatches.Patch(color='crimson',   label='Illicit (0)'),
        mpatches.Patch(color='steelblue', label='Licit (1)'),
        mpatches.Patch(color='lightgrey', label='Unknown (-1)'),
    ]
    ax2.legend(handles=legend_handles)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":

    dataset = EllipticDataset(root=processed_data_location)
    #visualize_elliptic_dataset(dataset, save_path=os.path.join(processed_data_location, 'elliptic_visualization.png'))
    #overlap = data[1].train_mask & data[1].test_mask
    #print(f"Overlapping nodes: {overlap.sum().item()}")
    #data = RealDataTraining(root=processed_data_path)
    print('test')
    #pre_process_ethereum()