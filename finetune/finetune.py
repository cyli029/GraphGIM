import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader.molcule_dataloader import MoleculeDataset
from torch_geometric.loader import DatalLoader
from utils.public_utils import setup_device, fix_train_random_seed, get_tqdm_desc, is_left_better_right

# graph
from model.graph_base.gnns import GINGraphPre, GINGraphRepr
from model.model_utils import get_classifier
from sklearn.metrics import roc_auc_score
from utils.splitter import scaffold_split
from train_utils import train_one_epoch, evaluate

def parse_args():
    # training setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0',help='GPUs of CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of gpus to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    # graph
    parser.add_argument('--nums_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    # dataset
    parser.add_argument('--dataroot', type=str, default='../dataset/downstream',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--dataset', type=str, default='bbbp',
                        help='the name of dataset. For now, only classification.')
    #train
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold.")
    parser.add_argument('--eval_train', type=int, default=1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--pretrained', action='store_true', help='loading the pretrained model or not')
    parser.add_argument('--resume', type=str, default='best_epoch=11_loss=2.40',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--task_type', type=str, default="classification", )
    args = parser.parse_args()
    return args

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device, device_ids = setup_device(args.ngpu)
    args.multigpu = False
    fix_train_random_seed(args.runseed)

    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    if args.task_type == 'classification':
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
            criterion = nn.L1Loss()
        else:
            eval_metric = "rmse"
            criterion = nn.MSELoss()
        valid_select = "min"
        min_value = np.inf
    else:
        raise Exception("param {} is not supported".format(args.task_type))

    print("eval_metric: {}".format(eval_metric))

    # set up dataset
    dataset = MoleculeDataset(os.path.join(args.dataroot,args.dataset), dataset=args.dataset)

    print(dataset)

    if args.split == 'scaffold':
        # './dataset/' + args.dataset + '/processed/success_smiles.csv'
        smiles_list = pd.read_csv(os.path.join(args.dataroot, args.dataset, 'processed/success_smiles.csv'), header=None)[
            0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,
                                                                    frac_valid=0.1, frac_test=0.1)
        print("finished scaffold split")
    else:
        raise ValueError("Invalid split option.")
    # loading data_loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GINGraphPre(num_layers=args.nums_layer,
                        emb_dim=args.emb_dim,
                        num_tasks=num_tasks,
                        JK=args.JK,
                        drop_ratio=args.dropout_ratio,
                        graph_pooling=args.graph_pooling)

    if args.pretrained:
        print('loading pretrained model')
        model._from_pretrained('../logs/pretraining/ckpts/multi_scale/{}.pth'.format(args.resume))
    model.to(device)

    # Loss and optimizer
    optimizer_params = [{"params": model.parameters()}]
    optimizer = optim.Adam(optimizer_params, lr=args.lr, weight_decay=args.decay)

    # train
    results = {
        'highest_valid': min_value,
        'final_test': min_value,
    }

    for epoch in range(1, args.epochs + 1):
        tqdm_train_desc, _, tqdm_eval_val_desc, tqdm_eval_test_desc = get_tqdm_desc(args.dataset, epoch)
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer, data_loader=train_loader, device=device, criterion=criterion,
                                     epoch=epoch, task_type=args.task_type, tqdm_desc=tqdm_train_desc)

        val_loss, val_results = evaluate(model=model,
                                         data_loader=val_loader, device=device, criterion=criterion,
                                         task_type=args.task_type, tqdm_desc=tqdm_eval_val_desc)

        test_loss, test_results = evaluate(model=model,
                                           data_loader=test_loader, device=device, criterion=criterion,
                                           task_type=args.task_type, tqdm_desc=tqdm_eval_test_desc)

        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]
        print({"dataset": args.dataset, "epoch": epoch, "Train Loss": train_loss, 'Validation': valid_result,
               'Test': test_result})
        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_test'] = test_result

        print("final results: {}".format(results))
    return results['final_test']

if __name__ == '__main__':
    args = parse_args()
    main(args=args)