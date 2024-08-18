import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import sys
import os
import torch
from tqdm import tqdm
from utils.evaluate import metric
from utils.evaluate import metric_multitask
from utils.evaluate import metric_reg
from utils.evaluate import metric_reg_multitask
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


def cal_downstream_loss(y_logit, labels, criterion, task_type):
    if task_type == 'classification':
        is_valid = labels != -1
        loss_mat = criterion(y_logit.double(), labels)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
    elif task_type == "regression":
        loss = criterion(y_logit.double(), labels)
    return loss


def train_one_epoch(graph_backbone, graph_predictor, optimizer, data_loader, criterion, device, epoch,
                    task_type, tqdm_desc=""):
    assert task_type in ['classification', 'regression']
    graph_backbone.train()
    graph_predictor.train()

    accu_loss = torch.zeros(1).to(device)
    accu_g_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc, ncols=160)
    for step, data in enumerate(data_loader):
        graphs = data
        graphs = graphs.to(device)
        sample_num += graphs.batch_size
        if "GINGraphRepr" in str(type(graph_backbone)):
            graph_representation = graph_backbone(graphs)
        else:
            node_representation = graph_backbone(graphs)
            graph_representation = global_mean_pool(node_representation, graphs.batch)

        y_logit_graph = graph_predictor(graph_representation)
        labels = graphs.y.view(y_logit_graph.shape).to(torch.float64)

        # calculate loss
        g_loss = cal_downstream_loss(y_logit_graph, labels, criterion, task_type)

        loss = g_loss
        loss.backward()
        accu_loss += loss.detach()
        accu_g_loss += g_loss.detach()
        data_loader.desc = ("[train epoch {}] loss: {:.3f}; g_loss: {:.3f};".
                            format(epoch, accu_loss.item() / (step + 1), accu_g_loss.item() / (step + 1)))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(graph_backbone, graph_predictor, data_loader, criterion, device,
             task_type="classification", tqdm_desc="", ret_index=False):
    assert task_type in ['classification', 'regression']
    graph_backbone.eval()
    graph_predictor.eval()

    accu_loss = torch.zeros(1).to(device)
    y_scores, y_true, y_pred, y_prob = [], [], [], []
    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc, ncols=160)
    for step, data in enumerate(data_loader):
        if ret_index:
            _, graphs = data
        else:
            graphs = data
        graphs = graphs.to(device)
        sample_num += graphs.batch_size

        with torch.no_grad():
            if "GINGraphRepr" in str(type(graph_backbone)):
                graph_representation = graph_backbone(graphs)
            else:
                node_representation = graph_backbone(graphs)
                graph_representation = global_mean_pool(node_representation, graphs.batch)

            y_logit_graph = graph_predictor(graph_representation)
            labels = graphs.y.view(y_logit_graph.shape).to(torch.float64)
            if task_type == "classification":
                is_valid = labels != -1
                loss_mat = criterion(y_logit_graph.double(), labels)
                loss_mat = torch.where(is_valid, loss_mat,
                                       torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            elif task_type == "regression":
                loss = criterion(y_logit_graph.double(), labels)
            accu_loss += loss.detach()
            data_loader.desc = "{}; loss: {:.3f}".format(tqdm_desc, accu_loss.item() / (step + 1))

        y_true.append(labels.view(y_logit_graph.shape))
        y_scores.append(y_logit_graph)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            return accu_loss.item() / (step + 1), metric(y_true, y_pred, y_pro, empty=-1)
        elif task_type == "regression":
            return accu_loss.item() / (step + 1), metric_reg(y_true, y_scores)
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            return accu_loss.item() / (step + 1), metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
        elif task_type == "regression":
            return accu_loss.item() / (step + 1), metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1])
    else:
        raise Exception("error in the number of task.")
