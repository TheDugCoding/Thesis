import networkx as nx
import pandas as pd
import os


from utils import get_structural_info

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path  = '../../Data/Processed/'
processed_data_location = os.path.join(script_dir, relative_path)

'''---AML_world dataset preprocessing---'''

def pre_process_aml_world():
    # Check if AMl world has already been preprocessed
    if not os.path.exists(os.path.join(processed_data_location, 'aml_world.graphml')):
        # Load the dataset
        df_aml_world = pd.read_csv('../../Data/Raw/AML_world/Small_LI/formatted_transactions.csv')

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
        df_rabobank = pd.read_csv('../../Data/Raw/rabobank/rabobank_data.csv', delimiter=';')

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
        df_saml_d = pd.read_csv('../../Data/Raw/SAML-D/SAML-D.csv')

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
        df_addr_addr = pd.read_csv('../../Data/Raw/Elliptic++_Dataset/AddrAddr_edgelist.csv')

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