import numpy as np
import pandas as pd
import os.path as osp
from itertools import repeat
from ogb.utils.mol import smiles2graph
from utils.splitter import scaffold_split
from torch_geometric.data import Data, DataLoader, InMemoryDataset


class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, dataset, raw_dirname='raw', transform=None, pre_transform=None,
                pre_filter=None,
                smiles_filename=None,
                empty=False,
                taks_type='classification'):
        self.root = root
        self.dataset = dataset
        self.raw_dirname = raw_dirname
        self.smiles_filename = smiles_filename
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths)
        self.total = len(self)
        self.taks_type = taks_type
    
    @property
    def raw_file_names(self):
        if self.smiles_filename is None:
            raw_file_name = os.path.join(self.raw_dir, "{}.csv".format(self.dataset))
        else:
            raw_file_name = os.path.join(self.raw_dir, self.smiles_filename)
        
        return [os.path.split(raw_file_name)[1]]
    
    @property
    def raw_dir(self):
        return osp.join(self.root, self.raw_dirname)
    
    @property
    def processed_file_names(self):
        # A list of files in the processed_dir which needs to be found in order to skip the processing.
        return "geometric_data_processed.pt"
    
    def download(self):
        # Downloads raw data into raw_dir
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        assert len(self.raw_file_names) == 1
        raw_file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        df = pd.read_csv(raw_file_path)
        columns = df.columns.tolist()
        assert 'smiles' in columns and 'label' in columns
        index = df.index.tolist()
        smiles, label = df["smiles"].values, self.get_label(df)
        # Read data into huge `Data` list.

        data_list = []
        success_data_smiles_list = []
        error_data_smiles_list = []
        for i, s, l in zip(index, smiles, label):
            try:  # error will be occurred when mol is None in mol.GetAtoms()
                print(f'label: {l}')
                graph_dict = smiles2graph(s)
                edge_attr = torch.tensor(graph_dict["edge_feat"], dtype=torch.long)
                edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
                x = torch.tensor(graph_dict["node_feat"], dtype=torch.long)
                y = torch.tensor(l, dtype=torch.long).view(1, -1)
                graph = Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, index=i)
                data_list.append(graph)
                success_data_smiles_list.append(s)
            except:
                error_data_smiles_list.append(s)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in bbbp paths
        for type, data_smiles_list in [("success", success_data_smiles_list), ("error", error_data_smiles_list)]:
            data_smiles_series = pd.Series(data_smiles_list)
            print(self.processed_dir)
            data_smiles_series.to_csv(os.path.join(self.processed_dir, '{}_smiles.csv'.format(type)), index=False,
                                      header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_label(self, df):
        return np.array(df["label"].apply(lambda x: np.array(str(x).split(" ")).astype(int).tolist()).tolist())

