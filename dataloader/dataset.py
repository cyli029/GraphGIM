import os
import torch
from itertools import repeat
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch, InMemoryDataset
from torch.utils.data.dataloader import default_colalte
from torchvision import transforms

import random
import numpy as np
from tqdm import tqdm
from dataloader.dual_data_utils import load_dual_aligned_data
import collectiions.abc as container_abcs

string_classes, int_classes = str, int

class DualCollater(object):
    def __init__(self, follow_batch, multigpu=False):
        self.follow_batch = follow_batch
        self.multigpu = multigpu

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            if self.multigpu:
                return batch
            else:
                return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, np.ndarray):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)

class AlignVisionGraphDataset(Dataset):
    def __init__(self, dataroot, dataset, image_folder_name, img_transformer=None, img_normalize=None, idx_imaged3d_list=[0, 1, 2, 3],
                 verbose=False, args=None):
        self.args = args
        self.data_dict = load_dual_aligned_data(dataroot, dataset, image_folder_name)
        self.total = len(self.data_dict['index'])
        self.img_transformer = img_transformer
        if args is not None:
            random.seed(args.seed)
            
    def get_image3d(self, index):
        view_path_list = self.data_dict['image3d_path_list'][index]
        image3d = [Image.open(view_path).convert('RGB') for view_path in view_path_list]
        if self.img_transformer is not None:
            image3d = list(map(lambda img: self.img_transformer(img).unsqueeze(0), image3d))
            image3d = torch.cat(image3d)
        return image3d

    def get_graph(self, index):
        graph_data = self.data_dict['graph_data'].__class__()
        if hasattr(graph_data, '__num_nodes__'):
            graph_data.num_nodes = self.data_dict["graph_data"].__num_nodes__[index]
        for key in self.data_dict["graph_data"].keys():
            item, slices = self.data_dict["graph_data"][key], self.data_dict["graph_slices"][key]
            start, end = slices[index].item(), slices[index + 1].item()
            # print(slices[index], slices[index + 1])
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data_dict["graph_data"].__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            graph_data[key] = item[s]

        return graph_data

    def get_diff_levels_features(self, index):
        features_list = []
        for i in range(1, 5):
            key = f'image_feature_list_{i}'
            features_list.append(self.data_dict[key][index])
        return features_list

    def __getitem__(self, item_index):
        img_data = self.get_image3d(item_index)
        graph_data = self.get_graph(item_index)
        feature_dict = self.get_diff_levels_features(item_index)
        label = self.data_dict["label"][item_index]
        return img_data, graph_data, feature_dict, label

    def get_batch_by_item_index(self, batch_item_index):
        img_data = []
        graph_data = []
        for item_index in batch_item_index:
            img = self.get_image3d(item_index)
            graph = self.get_graph(item_index)
            img_data.append(img)
            graph_data.append(graph)
        return img_data, graph_data

    def __len__(self):
        return self.total