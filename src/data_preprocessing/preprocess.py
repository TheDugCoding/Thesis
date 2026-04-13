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

'''---Elliptic++ dataset preprocessing---'''


def pre_process_elliptic():

    # variable used to check for missing graphs
    graphs = []
    df_edges = None
    df_features = None

    # we use the first 42 timestamps because in later timestamps there are disruptions
    for graph_timestep in range(1, 43):

        graphml_path = os.path.join(processed_data_location, f'elliptic_addr_addr_timestep_{graph_timestep-1}.graphml')

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

        # PyG datasets need an x as for node features, we add a dummy feature to make them work
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
        for graph in graphs[35:42]:
            test_node_ids.update(graph.nodes())

        for i in range(35):
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

        # compute mean and std from graphs 0-34 only to avoid data leakage from test graphs (35-41)
        x_train = torch.cat([d.x for d in data_list[:35]], dim=0)
        topo_train = torch.cat([d.topological_features for d in data_list[:35]], dim=0)

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
#               data.train_mask = labeled_mask
            elif i < 35:
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

        # standardize topological features (z-score)
        topo_mean_rabo = topological_features_rabo.mean(dim=0)
        topo_std_rabo = topological_features_rabo.std(dim=0)
        # avoid dividing by zero
        topo_std_rabo[topo_std_rabo == 0] = 1
        topological_features_rabo = (topological_features_rabo - topo_mean_rabo) / topo_std_rabo

        # Create and save the PyG Data object, in future add the edge features if required
        data_rabo = Data(x=x_rabo, edge_index=pyg_aml_rabobank.edge_index, edge_attr=pyg_aml_rabobank.edge_attr, topological_features=topological_features_rabo)
        node_transform = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.2)
        data_rabo = node_transform(data_rabo)

        # add a new variable for the topological features
        topological_features_ethereum = pyg_ethereum.x[:, [1, 2, 3, 4]]
        # store the remaining
        x_ethereum = pyg_ethereum.x[:, [0]]

        # standardize topological features (z-score)
        topo_mean_ethereum = topological_features_ethereum.mean(dim=0)
        topo_std_ethereum = topological_features_ethereum.std(dim=0)
        # avoid dividing by zero
        topo_std_ethereum[topo_std_ethereum == 0] = 1
        topological_features_ethereum = (topological_features_ethereum - topo_mean_ethereum) / topo_std_ethereum

        # Create and save the PyG Data object, in future add the edge features if required
        data_ethereum = Data(x=x_ethereum, edge_index=pyg_ethereum.edge_index,
                         edge_attr=pyg_ethereum.edge_attr, topological_features=topological_features_ethereum)
        node_transform = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.2)
        data_ethereum = node_transform(data_ethereum)

        # add a new variable for the topological features
        topological_features_ecr_20 = pyg_ecr_20.x[:, [ 1, 2, 3, 4]]
        # store the remaining
        x_ecr_20 = pyg_ecr_20.x[:, [0]]

        # standardize topological features (z-score)
        topo_mean_ecr_20 = topological_features_ecr_20.mean(dim=0)
        topo_std_ecr_20 = topological_features_ecr_20.std(dim=0)
        # avoid dividing by zero
        topo_std_ecr_20[topo_std_ecr_20 == 0] = 1
        topological_features_ecr_20 = (topological_features_ecr_20 - topo_mean_ecr_20) / topo_std_ecr_20

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

    dataset = RealDataTraining(root=processed_data_location)

    #dataset = EllipticDataset(root=processed_data_location)
    #visualize_elliptic_dataset(dataset, save_path=os.path.join(processed_data_location, 'elliptic_visualization.png'))
    #overlap = data[1].train_mask & data[1].test_mask
    #print(f"Overlapping nodes: {overlap.sum().item()}")
    #data = RealDataTraining(root=processed_data_path)
    print('test')
    #pre_process_ethereum()