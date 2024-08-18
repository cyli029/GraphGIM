import os
import torch
import numpy as np
import pandas as pd
from dataloader.data_utils import load_image3d_data_list
from train.procee_data.generators_features import save_load_multi_image3d_features


def process_smiles(smiles_file_path):
    xxx
    return True

def load_dual_data_filepath(dataroot, dataset, img_folder_name):
    processed_root = os.path.join(dataroot, dataset, 'processed')
    image_folder_path = os.path.join(processed_root, img_folder_name)
    graph_file_path = os.path.join(processed_root, 'geometric_data_processed.pt')
    processed_file_path = f'{processed_root}/{dataset}_sucess_smiles.csv'
    return image_folder_path, graph_file_path, processed_file_path


def get_label_from_align_data(label_series, task_type="classification"):
    '''e.g. get_label_from_align_data(df["label"])'''
    if task_type == "classification":
        return np.array(label_series.apply(lambda x: np.array(str(x).split(" ")).astype(int).tolist()).tolist())
    elif task_type == "regression":
        return np.array(label_series.apply(lambda x: np.array(str(x).split(" ")).astype(float).tolist()).tolist())
    else:
        raise UserWarning("{} is undefined.".format(task_type))
        
        
def load_image3d_data_list(dataroot, dataset, image3d_type="processed", label_column_name="label",
                           image3d_dir_name="image3d", csv_suffix="", is_cache=False, logger=None):
    log = print if logger is None else logger.info
    cache_path = f"{dataroot}/{dataset}/{image3d_type}/cache_{dataset}_load_image3d_data_list.pkl"
    if is_cache:
        if os.path.exists(cache_path):
            log(f"load from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data["image3d_index_list"], data["image3d_path_list"], data["image3d_label_list"]

    csv_file_path = os.path.join(dataroot, dataset, image3d_type, "{}{}.csv".format(dataset, csv_suffix))
    df = pd.read_csv(csv_file_path)
    columns = df.columns.tolist()
    assert "index" in columns and label_column_name in columns
    image3d_root = f"{dataroot}/{dataset}/{image3d_type}/{image3d_dir_name}"

    image3d_index_list = df["index"].tolist()
    image3d_label_list = df[label_column_name].apply(lambda x: str(x).split(' ')).tolist()
    image3d_path_list = []

    for image3d_index in tqdm(image3d_index_list, desc="load_image3d_data_list"):
        image3d_path = []
        for filename in os.listdir(f"{image3d_root}/{image3d_index}"):
            if filename.endswith('.png'):
                image3d_path.append(f"{image3d_root}/{image3d_index}/{filename}")
        image3d_path_list.append(image3d_path)

    if not is_cache:
        log(f"save to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "image3d_index_list": image3d_index_list,
                "image3d_path_list": image3d_path_list,
                "image3d_label_list": image3d_label_list},
                f)
    return image3d_index_list, image3d_path_list, image3d_label_list


def check_processed_dataset(dataroot, dataset, img_folder_name):
    image_folder_path, graph_file_path, processed_file_path = load_dual_data_filepath(dataroot, dataset, img_folder_name)
    print(f'loading data file image_folder_path; graph_file_path')
    if not (os.path.exists(image_folder_path) and os.path.exists(graph_file_path) and os.path.exists(processed_file_path)):
        return False
    
    df = pd.read_csv(processed_file_path)
    cols = df.columns
    if not ("index" in cols and "smiles" in cols and "label" in cols):
        return False
    index = df['index'].vlaues.astype(str).tolist()
    
    # check aligned graph and image
    graph_data = torch.load(graph_file_path)
    graph_index = [str(item) for item in graph_data[0].index]
    
    image3d_index_list, image3d_path_list, image3d_label_list = load_image3d_data_list(dataroot=dataroot, dataset=dataset,is_cache=False,csv_suffix='_success_smiles')
    image3d_index_list = [str(item) for item in image3d_index_list]
    
    if not len(index) == len(graph_index):
        raise ValueError('existed invalid smiles can not tranfer into graph')
        return False
    elif not len(index) == len(image3d_index_list):
        raise ValueError('existed invalid image3d_index_list')
        return False
    return True

def load_dual_alinged_data(dataroot, dataset, img_folder_name, task_type='classification', verbose=False):
    if not check_processed_dataset(dataroot, dataset, img_folder_name):
        raise ValueError('dataset initial failed...')
    
    img_folder, graph_path, processed_file_path = load_dual_data_filepath(dataroot,dataset, img_folder_name)
    
    df = pd.read_csv(processed_file_path)
    index = df['index'].astype(str).tolist()
    _index = df['index'].astype(str).values
    # load graph data
    graph_data, graph_slices = torch.load(graph_path)
    graph_index = np.array(graph_data.index).astype(str).tolist()
    
    label = get_label_from_align_data(df['label']m task_type=task_type)
    
    image3d_index_list, image3d_path_list, image_label_list = load_image3d_data_list(dataroot=dataroot, dataset=dataset, is_cache=True)
    
    # load different scales features
    index_list, image_feature_layer1_list, image_feature_layer2_list, image_feature_layer3_list, image_feature_layer4_list = save_load_multi_image3d_features(dataroot=dataroot, dataset=dataset, is_cache=True, ret_index=True)
    
    index_list = np.array(index_list).astype(str).tolist()
    assert ((np.array(graph_index) == np.array(index_list)).all() == (np.array(_index) == np.array(index_list)).all()), "index from graph  and index from csv file is inconsistent"
    return {
        'index':_index,
        'label':label,
        'graph_data': graph_data,
        'graph_slices': graph_slices,
        'image3d_path_list' = image3d_path_list,
        'image_feature_layer1_list' = image_feature_layer1_list,
        'image_feature_layer2_list' = image_feature_layer2_list,
        'image_feature_layer3_list' = image_feature_layer3_list,
        'image_feature_layer4_list' = image_feature_layer4_list 
    }