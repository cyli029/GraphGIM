import os
import argparse 
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.image_encoder import Restnet18
from utils.public_utils import setup_device
from dataloader.image_dataloader import ImageDataset
from dataloader.data_utils import transforms_for_train_aug
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


'''
    the feature map of the resnet18
        feature_map_list : [...]
        item_0 : layer_1_model = nn.Sequential(*list(model.children())[:5])
        item_1:  layer_2_model = nn.Sequential(*list(model.children())[:6])
        item_2:  layer_3_model = nn.Sequential(*list(model.children())[:7])
        item_3:  layer_4_model = nn.Sequential(*list(model.children())[:8])
'''

def process_multi_images(image3d):
    n_samples, n_views, n_channels, h, w = image3d.shape
    return n_samples, n_views, image3d.reshape(n_samples * n_views, n_channels, h, w)

def process_scales_features(image_feature, n_samples, n_views, level_type='level1'):
    assert level_type in ['level1', 'level2', 'level3', 'level4']
    image_channels = image_feature.size(1)
    if level_type == 'level1':
        w, h = 56, 56
    elif level_type == 'level2':
        w, h = 28, 28
    elif level_type == 'level3':
        w, h = 14, 14
    elif level_type == 'level4':
        w, h = 7, 7
    else:
        raise ValueError('Invalid level type')
    image_feature = image_feature.reshape(n_samples, n_views, image_channels, w, h)
    mean_image_feature_multi = image_feature.mean(dim=1, keepdim=False)
    pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    image_feature_pool = pool(mean_image_feature_multi)
    pooled_features = image_feature_pool.view(image_feature_pool.size(0), -1)
    return pooled_features


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetFeatureExtractor, self).__init__()
        # get the different level feature of  resnet_model
        self.level1 = nn.Sequential(*list(resnet_model.children())[:5])
        self.level2 = nn.Sequential(*list(resnet_model.children())[5])
        self.level3 = nn.Sequential(*list(resnet_model.children())[6])
        self.level4 = nn.Sequential(*list(resnet_model.children())[7])

    def forward(self, x):
        n_samples, n_views, x = process_image(x)
        with torch.no_grad():
            layer_dict_output = {'level1': self.layer1(x)}
            layer_dict_output['level2'] = self.layer2(layer_dict_output['level1'])
            layer_dict_output['level3'] = self.layer3(layer_dict_output['level2'])
            layer_dict_output['level4'] = self.layer4(layer_dict_output['level3'])
        feature_level_list = []
        for key, item in layer_dict_output.items():
            feature_diff_level = process_scales_features(image_feature=item, n_samples=n_samples, n_views=n_views,
                                                     level_type=key)
            feature_level_list.append(feature_diff_level.cpu())
        return feature_level_list

def process_image3d_features(model, data_loader, device):
    data_loader = tqdm(data_loader, total=len(data_loader), ncols=160)
    image_feature_layer1_list = []
    image_feature_layer2_list = []
    image_feature_layer3_list = []
    image_feature_layer4_list = []
    index_list = []
    for step, data in enumerate(data_loader):
        extractor = ResNetFeatureExtractor(resnet_model=model).to(device)
        data, index = data
        data = data.to(device)
        index_list.extend(index.cpu().numpy())
        batch_diff_features = extractor(data)
        for level_idx, features in enumerate(batch_diff_features):
            if level_idx == 0:
                image_feature_layer1_list.extend(features)
            elif level_idx == 1:
                image_feature_layer2_list.extend(features)
            elif level_idx == 2:
                image_feature_layer3_list.extend(features)
            else:
                image_feature_layer4_list.extend(features)
    return index_list, image_feature_layer1_list, image_feature_layer2_list, image_feature_layer3_list, image_feature_layer4_list

def save_load_multi_image3d_features(dataroot, dataset, transform=None, ret_index=False, batch_size=256, num_works=4,
                                   extrator=None,is_cache=False, logger=None, type='multi_scale_feature'):
    idx_features_list = [0, 1, 2, 3]
    log = print if logger is None else logger.info
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
    suffix = '_'.join([str(item) for item in idx_features_list])
    cache_data_path = f'./cache/Process_{type}@{dataset}@{suffix}.npz'
    if os.path.exists(cache_data_path) and is_cache:
        log(f'loading cached {type} from {cache_data_path}')
        data = np.load(cache_data_path)
        if ret_index:
            return data['index_list'], data['multi_scale_feature_level1'], data['multi_scale_feature_level2'],
        data['multi_scale_feature_level3'], data['multi_scale_feature_level4']
        else:
            return  data['multi_scale_feature_level1'], data['multi_scale_feature_level2'],
        data['multi_scale_feature_level3'], data['multi_scale_feature_level4']
    else:
        image3d_dataset = ImageDataset(dataroot=dataroot, dataset=dataset, transform=transform,ret_index=True)
        
        image3d_dataloader = DataLoader(dataset = image3d_dataset, batch_size = batch_size, num_works = num_works,
                                       shuffle=False, pin_memory=True)
        
        device, device_ids = setup_device(1)
        index_list, multi_scale_feature_level1, multi_scale_feature_level2,multi_scale_feature_level3,multi_scale_feature_level4 = (
        process_image3d_features(model=extractor, data_loader=image_dataloader, device=device))
        
        # save
        np.savez(cache_data_path, index_list=index_list, multi_scale_feature_level1=multi_scale_feature_level1,
                multi_scale_feature_level2=multi_scale_feature_level2, multi_scale_feature_level3=multi_scale_feature_level3,
                multi_scale_feature_level4=multi_scale_feature_level4)
        return 1

def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataroot', type=str, default='../../dataset/pre-training')
    parser.add_argument('--dataset', type=str, default='iem-200w')

    # model
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--resume', type=str, default='../../resumes/pretrained-model/IEM.pth')
    parser.add_argument('--resume_model_name', type=str, default='image3d_teacher')
    parser.add_argument('--type', type=str, default='process',choices=['process','load'])
    
    # dataloader
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_works', type=int, default=4)
    
    #save 
    parser.add_argument('--is_cache',action='store_true')
    
'''
def save_load_multi_image3d_features(dataroot, dataset, transform=None, ret_index=False, batch_size=256, num_works=4,
                                   extrator=None,is_cache=False, logger=None, type='multi_scale_feature'):
'''


def main(args):
    _transforms = transforms_for_train_aug(resize=224, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    feature_extractor = Restnet18(args.model_name, args.resume, args.resume_model_name).teacher.model
    if args.type == 'process'
        args.is_cache = False
        save_load_multi_image3d_features(dataroot=args.dataroot, dataset=arg.dataset, transform=_transforms,
                                        ret_index=False,batch_size=args.batch_size,num_works=args.num_works,
                                        extractor=feature_extractor,is_cache=args.is_cache)
    else:
        args.is_cache = True
        save_load_multi_image3d_features(dataroot=args.dataroot, dataset=arg.dataset,
                                        ret_index=True,is_cache=args.is_cache)

        
if __name__ == '__main__':
    args = parse_args()
    main(args)