import os
import pickle

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import from_networkx, subgraph
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


from src.data_preprocessing.utils import get_structural_info
from src.utils import get_data_sub_folder, get_data_folder

script_dir = get_data_folder()
relative_path_processed = 'processed'
processed_data_location = get_data_sub_folder(relative_path_processed)

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
    # Check if the AddrAddr graph has already been preprocessed
    if not os.path.exists(os.path.join(processed_data_location, 'elliptic_addr_addr.graphml')):
        # Load the dataset
        df_addr_addr = pd.read_csv(os.path.join(script_dir, relative_path_elliptic_raw_edges))
        df_addr_addr = df_addr_addr.drop_duplicates()
        df_wallet_features = pd.read_csv(os.path.join(script_dir, relative_path_elliptic_raw_node_features))
        #keep
        #df_wallet_features = df_wallet_features.drop_duplicates(subset='address', keep='last')

        # Keep address separately for later use
        addresses = df_wallet_features['address']
        classes = df_wallet_features['class']
        # Drop non-feature columns (like 'address' or 'time_step') if any
        feature_columns = df_wallet_features.columns.difference(['address','timestep','class'])
        features = df_wallet_features[feature_columns]
        # Normalize features
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features)
        df_wallet_features = pd.DataFrame(normalized_features, columns=feature_columns)
        df_wallet_features['address'] = addresses
        df_wallet_features['class'] = classes



        # Initialize a directed graph
        G_addr_addr = nx.DiGraph()

        # Add node attributes from df_wallet_features
        df_wallet_features = df_wallet_features.set_index('address')

        # Add edges to the graph from the dataset
        for index, row in tqdm(df_addr_addr.iterrows(), total=df_addr_addr.shape[0]):
            G_addr_addr.add_edge(row['input_address'], row['output_address'])

        a = 0
        b = 0
        c = 0

        for node in tqdm(G_addr_addr.nodes(), total=G_addr_addr.number_of_nodes()):
            if node in df_wallet_features.index:
                # Get all rows for the node
                node_data = df_wallet_features.loc[node]

                if isinstance(node_data, pd.DataFrame):
                    a = a + 1
                    attr_dict = node_data.iloc[-1].to_dict()
                elif isinstance(node_data, pd.Series):
                    attr_dict = node_data.to_dict()
                    b = b + 1
                else:
                    c = c + 1
                    print(f"Unexpected format for node {node}: {type(node_data)}")



                nx.set_node_attributes(G_addr_addr, {node: attr_dict})

        with open("delete.txt", "w") as f:
            print(a, b, c, file=f)
        print("setting node attributes")
        # Compute additional structural information
        G_addr_addr = get_structural_info(G_addr_addr)

        # Save dataset with additional information
        nx.write_graphml(G_addr_addr, os.path.join(processed_data_location, 'elliptic_addr_addr.graphml'))

    else:
        G_addr_addr = nx.read_graphml(os.path.join(processed_data_location, 'elliptic_addr_addr.graphml'))

    return G_addr_addr


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

        # Iterate through the dataset and add edges until 80,000 unique nodes are reached
        for _, row in df_erc20.iterrows():
            from_addr = row['from_address']
            to_addr = row['to_address']

            # Add dummy node feature if node not already present
            if from_addr not in G:
                G.add_node(from_addr, dummy=0)
            if to_addr not in G:
                G.add_node(to_addr, dummy=0)

            G.add_edge(from_addr, to_addr,
                             value=row['value'],
                             time_stamp=row['time_stamp'])


            # Stop if we've reached at least 80,000 unique nodes
            if G.number_of_nodes() >= 600000 or G.number_of_edges() >= 1200000:
                break

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
        return ['ellipticdataset_0.pt', 'ellipticdataset_1.pt', 'ellipticdataset_2.pt', 'ellipticdataset_3.pt', 'ellipticdataset_4.pt']

    def process(self):
        """Processes raw data into PyG data objects and saves them as .pt files."""
        # Generate the graph data from pre-processing functions


        pyg_elliptic = from_networkx(pre_process_elliptic(), group_node_attrs=[
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
        ])

        # add a new variable for the topological features
        topological_features = pyg_elliptic.x[:, [
                                  0, 1, 2, 3 ]]

        # store the remaining
        x = pyg_elliptic.x[:, [
                                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                  23,
                                  24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                  43, 44,
                                  45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
                              ]]
        y = (pyg_elliptic.x[:, 4] - 1).long()

        time_step = pyg_elliptic.x[:, 60]

        #pyg_elliptic.x = pyg_elliptic.x.float()

        # Create and save the PyG Data object, in future add the edge features if required
        data = Data(x=x, edge_index=pyg_elliptic.edge_index, topological_features = topological_features, y=y, time_step=time_step)
        node_transform = RandomNodeSplit(split="train_rest",num_val=0.1,num_test=0.2)
        data_original = node_transform(data)
        # the Elliptic dataset is very unbalanced so we create a balanced version
        data_balanced = RandomNodeSplit(split='random', num_train_per_class=14266, num_val=0.1, num_test=0.2)(data)

        """
        x_2class = x[mask_2class]
        topo_2class = topological_features[mask_2class]
        y_2class = y[mask_2class].long()
        edge_index_2class, _ = subgraph(mask_2class, pyg_elliptic.edge_index, relabel_nodes=True)
        """
        # creating a third version, this version only contain illicit '0' and licit '1' labels
        # Filter to keep only class 0 and class 1 (exclude class 2)
        mask_2class = y < 2  # keeping only class 0 and 1

        #keeping the dataset but only use licit and illicit nodes for training
        # create new Data object with filtered nodes
        data_2class = Data(x=x, edge_index=pyg_elliptic.edge_index,
                           topological_features=topological_features, y=y)

        num_known_nodes = mask_2class.sum().item()
        permutations = torch.randperm(num_known_nodes)
        train_size = int(0.8 * num_known_nodes)
        val_size = int(0.1 * num_known_nodes)
        test_size = num_known_nodes - train_size - val_size
        data_2class.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data_2class.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data_2class.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_indices = mask_2class.nonzero(as_tuple=True)[0][permutations[:train_size]]
        val_indices = mask_2class.nonzero(as_tuple=True)[0][permutations[train_size:train_size + val_size]]
        test_indices = mask_2class.nonzero(as_tuple=True)[0][permutations[train_size + val_size:]]


        data_2class.train_mask[train_indices] = True
        data_2class.val_mask[val_indices] = True
        data_2class.test_mask[test_indices] = True

        #creating a fourth class this version only contain illicit '0' and licit '1' labels,
        # and they have the same number of samples

        data_2class_balanced = data
        #replace class 2with -1, to unlabel them
        data_2class_balanced.y[data_2class_balanced.y == 2] = -1
        data_2class_balanced = RandomNodeSplit(split='random', num_train_per_class=11000, num_val=0.1, num_test=0.2)(data_2class_balanced)
        #data_2class_balanced = T.RemoveTrainingClasses([2])(data_2class_balanced)
        # unlabel class 2
        class_2_nodes = data_2class_balanced.y == -1
        data_2class_balanced.val_mask[class_2_nodes] = False
        data_2class_balanced.test_mask[class_2_nodes] = False

        torch.save(data, self.processed_paths[0])
        torch.save(data_original, self.processed_paths[1])
        torch.save(data_balanced, self.processed_paths[2])
        torch.save(data_2class, self.processed_paths[3])
        torch.save(data_2class_balanced, self.processed_paths[4])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Loads and returns the graph at the given index."""
        data = torch.load(os.path.join(self.processed_dir, f'ellipticdataset_{idx}.pt'))
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

if __name__ == "__main__":
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
    
    
    dataset = AmlSimDataset(root = processed_data_location)
    '''

    # dataset = RealDataTraining(root = processed_data_location, add_topological_features=True)
    #pre_process_ethereum()
    #pre_process_erc_20_stablecoin()
    #pre_process_elliptic()
    #relative_path_processed = 'processed'
    processed_data_path = get_data_sub_folder(relative_path_processed)
    #data = EllipticDataset(root=processed_data_path)
    data = RealDataTraining(root=processed_data_path)

    #pre_process_ethereum()