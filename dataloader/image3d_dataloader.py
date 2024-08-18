import os
import torch
import numpy as np
from PIL import Image
from utils.splitter import *
from torchvision impor transfroms
from torch.utils.data import Dataset, DataLoader
from dataloader.data_utils import check_num_of_image3d
from dataloader.dual_data_utils import load_image3d_data_list

class Image3dDataset(Dataset):
    def __init__(self, dataroot, dataset, split='all',transform=None, ret_index=None, args=None,
                idx_image3d_list=[0,1,2,3], logger=None):
        
         '''
         :param dataroot: '../dataset/pre-training'
         :param dataset: 'iem-200w'
         :param split: 'all'
         '''
        self.logger = logger
        self.log = print if logger is None else logger.info
        self.split = split
        assert self.split in ['train','val','test','all']
        
        # load data
        multi_vew = len(idx_image3d_list)
        self.multi_view = multi_vew
        index_list, image3d_path_list, label_list = self.load_3d_data(dataroot=dataroot,
                                                                     dataset=dataset,
                                                                    idx_image3d_list=idx_image3d_list)
        self.args = args
        self.index_list = index_list
        self.image3d_path_list = image3d_path_list
        self.label_list = label_list
        self.total = len(self.image3d_path_list)
        self.total_view = self.total * multi_vew
        self.transform = transform
        self.ret_index = ret_index
    def load_3d_data(self, dataroot, dataset, idx_image3d_list):
        if not os.path.exists('./cache'):
            os.makedirs('./cache')
        suffix = '-'.join([str(item) for item in idx_image3d_list])
        cache_data_path = f'./cache/PretrainDataset@{dataset}@{self.split}@{suffix}.npz'
        if os.path.exists(cache_data_path):
            self.log(f'loading cached data: {cache_data_path}')
            data = np.load(cache_data_path, allow_pickle=True)
            return data['index_list'], data['image3d_path_list'], data['label_list']
        else:
            index_list, image3d_path_list, label_list = load_image3d_data_list(dataroot, datasest=dataset,
                                                                              label_cloumn_name='label',
                                                                              is_cache=True,
                                                                              logger=self.logger)
            n_total = len(index_list)
            if self.split == 'train':
                index_list, image3d_path_list, label_list = (index_list[:int(n_total*0.8)],
                                                            image3d_path_list[:int(n_total*0.8)],
                                                            label_list[:int(n_total*0.8)])
            elif self.split == 'val':
                index_list, image3d_path_list, label_list = (index_list[:int(n_total*0.1)],
                                                            image3d_path_list[:int(n_total*0.1)],
                                                            label_list[:int(n_total*0.1)])
            elif self.split == 'test':
                index_list, image3d_path_list, label_list = (index_list[:int(n_total*0.1)],
                                                            image3d_path_list[:int(n_total*0.1)],
                                                            label_list[:int(n_total*0.1)])
            else:
                index_list, image3d_path_list, label_list = index_list, image3d_path_list, label_list
            
            tmp_image3d_path_list = []
            for tmp_list in tqdm(image3d_path_list, desc=f' check {idx_image3d_list}'):
                tmp_image3d_path_list.append(np.array(tmp_list)[idx_image3d_list].tolist())
                
            image3d_path_list = tmp_image3d_path_list
            multi_view = len(idx_image3d_list)
            check_num_of_image3d(image3d_path_list, multi_vew)
            #save to cache
            self.log(f'save cache to {cache_data_path}')
            np.savez(cache_data_path,index_list=index_list, image3d_path_list=image3d_path_list, label_list=label_list)
        return index_list, image3d_path_list, label_list
    
    def get_image3d(self, index):
        view_path_list = self.image3d_path_list[index]
        image3d = [Image.open(view_path_list).convert('RGB') for view_path in view_path_list]
        if self.transform is not None:
            image3d = list(map(lamba img: self.transform(img).unsqueeze(0), image3d))
            image3d = torch.cat(image3d)
        return image3d
    
    def __getitem__(self,index):
        image3d = self.get_image3d(index)
        if self.ret_index:
            return image3d, self.index_list[index]
        else:
            return image3d
    def __len__(self):
        return self.total