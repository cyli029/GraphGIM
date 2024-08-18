import os
import torch
import numpy as np
import pandas as pd
import os.path as osp
from utils import splitter
from ogb.utils.mol import smiles2graph
from torch_geometric.data import Data, InMemoryDataset

def mol2grpah(smiles):
    """
        using ogb to extract graph featrues
    """
    # trabfer the smiles to graph
    graph_dict = smiles2graph(smiles)
    edge_attr = torch.tensor(graph_dict['edge_feat'], dtype=torch.long)
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
    x = torch.tensor(graph_dict['node_feat'], dtype=torch.long)
    
    return x, edge_index, edge_attr


def create_graph_data(raw_file_path, dataset, save_root, pre_filter=None, pre_transform=None):
    '''
        the structure of output file:
            # save_root
                geometric_data_processed.pt
                {datasest}_error_smiles.csv
                {datasest}_success_smiles.csv
            raw_file_path: '../dataset/pre-training/iem-200w/'
    '''
    # process raw data and save it into the processed_dir
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        print(f'created dir {save_root}')
    df = pd.read_csv(raw_file_path)
    columns = df.columns.tolist()
    assert 'index' in columns and 'canonical_smiles' in columns and 'random_smiles' in columns and 'label' in columns
    index, smiles label = df['index'].values, df['canonical_smiles'].values, df['label'].values
    # read data into huge 'Data' list
    data_list = []
    success_data_smiles_list = []
    error_data_idx_list = []
    error_data_smiles_list = []
    for i, s, l in zip(index, smiles, label):
        try:
            graph_dict = smiles2graph(s)
            edge_attr = torch.tensor(graph_dict['edge_feat'], dtype=torch.long)
            edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
            x = torch.tensor(graph_dict['edge_feat'], dtype=torch.long)
            y = torch.tensor(l, dtype=torch.long).view(1,-1)
            graph = Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, index=i)
            data_list.append(graph)
            success_data_smiles_list.append((i, s, l))
        except:
            print(f'error {(i,s,l)}')
            error_data_smiles_list.append((i, s, l))
            error_data_idx_list.append(i)
    
    if pre_filter is not None:
        data_list = [data for data in data_list if pre_filter(data)]
    
    if pre_transform is not None:
        data_list = [pre_transform(data) for data in data_list]
    
    # write data_smiles_list in raw_paths
    for type, data_smiles_list in [('success', success_data_smiles_list), ('error', error_data_smiles_list)]:
        data_smiles_list_dataframe = pd.DataFrame(data_smiles_list, columns=['index','smiles','label'])
        data_smiles_list_dataframe.to_csv(os.path.join(save_root, '{}_{}_smiles.csv'.format(dataset, type)), index=False, header=True)
    
    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), os.path.join(save_root, 'geometric_data_processed.pt'))
    return error_data_smiles_list


def create_graph_from_smiles(smiles, label, index=None, pre_filter=None, pre_transform=None,
                             task_type="classification"):
    assert task_type in ["classification", "regression"]
    try:
        x, edge_index, edge_attr = mol2graph(smiles)
        if task_type == "classification":
            y = torch.tensor(label, dtype=torch.long).view(1, -1)
        else:
            y = torch.tensor(label, dtype=torch.float).view(1, -1)
        graph = Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, index=index)
        if pre_filter is not None and pre_filter(graph):
            return None
        if pre_transform is not None:
            graph = pre_transform(graph)
        return graph
    except:
        return None