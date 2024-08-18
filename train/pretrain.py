"""
-*- coding: utf-8 -*-
@Author: ChaoYiLi
@description: pretraining
@tools: @pycharm
@Time: 2024/05/20 16:17
"""
import os
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.public_utils import setup_device, fix_train_random_seed
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

#dataloader
from dataloader.data_utils import transforms_for_train_aug, transforms_for_eval
from dataloader.dataset import AlignVisionGraphDataset, DualCollater
#graph
from torch_geometric.nn import global_mean_pool
from train.model_utils import train_one_epoch
from model.image_encoder import Restnet18
from model.model_utils import save_checkpoint, write_result_dict_to_tb
from model.graph_base.gnns import GNN
#linear map
from model.base.base_utils import linear_map
#prompt
from model.base.prompt import prompt_generator
#loss
from loss.losses import SupConLoss

def normalize(x):
    return F.normalize(x, p=2, dim=1)'


class graphCL(nn.Module):
    def __init__(self, gnn):
        super(graphCL, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 512), nn.ReLU(inplace=True), nn.Linear(512, 512))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x_node = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x_node, batch)
        x = self.projection_head(x)
        x = normalize(x)
        return x_node, x
    
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch pretraining dual graph & 3dimage')
    
    #basic
    parser.add_argument('--dataroot',type=str, default='../dataset/pre-training')
    parser.add_argument('--dataset', type=str, default='iem-200w', help='e.g. iem-1w, iem-200w')
    parser.add_argument('--image_dir_name', type=str, default='image3d')
    parser.add_argument('--gpu', type=str, default='0',help='GPUs of CUDA_VISIBLE_DEVICES, e.g 0,1,2,3')
    parser.add_argument('--ngpu', type=int, default=1,help='number of gpus to use')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    
    #gnn
    parser.add_argument('--num_laeyrs', type=int, default=5,help='the number of layers of Gin')
    parser.add_argument('--feat_dim', type=int, default=300, help='the dimension of topological space')
    parser.add_argument('--JK', type=str, default='last', choices=['concat','last','max','sum'])
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio of gin')
    parser.add_argument('--graph_pooling', type=str, default='mean', help='the pooling type of graph')
    
    #3Dimage
    parser.add_argument('--resume',type=str,help='resume training from a path of checkpoint')
    parser.add_argument('--resume_model_name',type=str,help='resume training from a path of checkpoint')
    
    #optimizer
    parser.add_argument('--lr',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-weight_decay', type=float, default=0, help='weight decay')
    
    #loss
    parser.add_argument('--temperature', type=float, default=0.05, help='loss temperature')
    parser.add_argument('--base_temperature', type=float, default=0.05, help='loss temperature')
    
    #train
    parser.add_argument('--seed',type=int,default=42,help='random seed (default:42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2024, help='random seed to run model (default: 2024)')
    parser.add_argument('--epochs', type=int, default=100, help='the train epochs')
     parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--model_name', type=str, default="resnet18", help='model name')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')
    
    #liner
    parser.add_argument('--embed_dim', type=int, default=512, help='embedding dimension')
    
    #prompt
    parser.add_argument('--prompt', action='store_true', help='using prompt or not')
    #multi-scale
    parser.add_argument('--multi_scale', action='store_true', help='using multi-scale contrastive learning')
    
    #log
    parser.add_argument('--log_dir', default='../logs/pretraining/', help='path to log')
    parser.add_argument('--tb_dir', default='../logs/tb', help='path to tensorboard logs')
    return parser.parse_args()


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # gpu
    device, device_ids = setup_device(args.ngpu)
    args.multigpu = False
    # fixed random seed
    fix_train_random_seed(args.runseed)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    train_transforms = transforms_for_train_aug(resize=arg.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD))
    
    train_dataset = AlignVisionGraphDataset(dataroot=args.dataroot,
                                           dataset=args.dataset,
                                           image_folder_name=args.image_dir_name,
                                           img_transformer=train_transforms)
    
    batch_size = args.batch
    dataloader = DataLoader(daaset=train_dataset,batch_size=batch_size,
                           num_workers=arg.workers,shuffle=True,pin_memory=True,collate=DualCollater(follow_batch=[], multigpu=False))
    
    # loaing resnet18 and initializing training setting
    image_encoder = Restnet18(modeL_name=args.model_name,
                             resume_teacher=args.resume,
                             resume_teacher_name=args.resume_model_name)
    # loading graph 
    graph = GNN(num_layers=args.num_layers, emb_dim=args.feat_dim, JK=args.JK, drop_ratio=args.drop_ratio)
    model = graphCL(graph)
    linear_prdictiors = None
    prompt = None
    
    if args.prompt and args.mutli_scale==False:
        inner_dim_list = [64, 128, 256, 512]
        prediction_heads = linear_map(inner_dim_list, embed_dim=args.embed_dim, size=len(inner_dim_list))
        args.hiddeb_size, arg.embedding_size = args.embed_dim, args.embed_dim
        prompt = prompt_generator(args)
        optimizer_params= [
            {'params': image_encoder.parameters()},
            {'params': model.parameters()},
            {'params': prediction_heads.parameters()},
            {'params': prompt.parameters()}
        ]
    elif args.multi_scale:
        inner_dim_list = [64, 128, 256, 512]
        prediction_heads = linear_map(inner_dim_list, embed_dim=args.embed_dim, size=len(inner_dim_list))
        optimizer_params = [
            {'params': image_encoder.parameters()},
            {"params": model.parameters()},
            {'params': prediction_heads.parameters()},
        ]
    else:
        optimizer_params = [
            {'params': image_encoder.parameters()},
            {"params": model.parameters()}
        ]
    
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    
    #loss
    criterion = SupConLoss(temperature=args.temperature, base_temperature=args.base_temperature)
    
    #lr scheduler
    lr_scheduler = None
    
    #initialize SummaryWriter from tensorboard.
       tb_writer = SummaryWriter(log_dir=args.tb_dir)
    optimizer_dict = {"optimizer": optimizer}
    #train
    best_loss = np.Inf
    
    for epoch in range(args.start_epoch, args.epochs):
        train_dict = train_one_epoch(image3d_encoder=image_encoder,
                                    graph_enocoder=model,
                                    linear_predictor=prediction_heads,
                                    prompt=prompt,
                                    multi_scale=args.multi_scale,
                                    optimizer=optimizer,
                                    data_loader=data_loader,
                                    criterion=criterion,
                                    device=device,
                                    epoch=epoch,
                                    args=args)
        if epoch==0:
            print(train_dict)
            
        #save model
        model_dict = {'image3d_encoder':image_encoder,'graph':model.gnn}
        
        if args.multi_scale and args.prompt ==False :
            model_dict['prediction_heads'] = prediction_heads
        elif arg.prompt:
            model_dict['prediction_heads'] = prediction_heads
            model_dict['prompt'] = prompt
        
        lr_scheduler_dict = {'lr_scheduler': lr_scheduler} if lr_scheduler is not None else None
        
        cur_loss = train_dict['avg_loss']
        
        if cur_loss < best_loss:
            train_type = ''
            if args.prompt:
                train_type = 'prompt'
                best_pre = "{}_best_epoch={}_loss={:.2f}".format(train_type, epoch, cur_loss)
            elif args.multi_scale:
                train_type = 'multi_scale'
                best_pre = "{}_best_epoch={}_loss={:.2f}".format(train_type, epoch, cur_loss)
            else:
                best_pre = "best_epoch={}_loss={:.2f}".format(epoch, cur_loss)
            if train_type == '':
                save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict,
                                train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                                name_pre=best_pre, name_post="")
            else:
                save_path = os.path.join(args.log_dir, "ckpts", train_type)
                save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict,
                                train_dict, epoch, save_path=save_path,
                                name_pre=best_pre, name_post="")
        
        write_result_dict_to_tb(tb_writer, train_dict, optimizer_dict)
        
if __name__ == '__main__':
    args = parse_args()
    main(args)