import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform

def transforms_for_train_aug(resize=224, mean_std=None, p=0.2):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.CenterCrop(resize), transforms.RandomHorizontalFlip(),
                                         transforms.RandomGrayscale(p), transforms.RandomRotation(degrees=360),
                                         transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def transforms_for_train(config, is_training=True):
    return create_transform(**config, is_training=is_training)


def transforms_for_eval(resize=(224, 224), img_size=(224, 224), mean_std=None):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def check_num_of_image3d(image3d_path_list, n_view):
    for idx, view_path_list in enumerate(image3d_path_list):
        assert len(view_path_list) == n_view, \
        'The view number of image3d {} is {}, not equal to the expected{}'\
        .format(idx, len(view_path_list), n_view)


