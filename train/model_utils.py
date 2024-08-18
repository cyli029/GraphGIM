import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np


# warm up learning rate scheduler
class warmup_scheduler:
    def __init__(self, optimizer, warmup_steps, initial_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.initial_lr * min(1.0, self.current_step / self.warmup_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# train one epoch
def train_one_epoch(image3d_encoder, graph_encoder, linear_predictor,
                    optimizer, data_loader, criterion, device, epoch, prompt=None, multi_scale=None, args=None):
    # n_sub_ckpt_list_step = ((np.arange(1, args.n_sub_checkpoints_each_epoch + 1) / (
    # args.n_sub_checkpoints_each_epoch + 1)) * len(data_loader)).astype(int)
    # imgage encoder graph encoder loading to cpu
    image3d_encoder.eval()
    graph_encoder.train()

    if linear_predictor is not None:
        linear_predictor.train()
        linear_predictor = linear_predictor.to(device)

    accu_loss = torch.zeros(1).to(device)
    accu_prompt_loss = torch.zeros(1).to(device)
    accu_multi_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, total=len(data_loader), ncols=160)

    for step, data in (enumerate(data_loader)):
        image3d, graph, feature_list, label = data
        image3d_feature_1, image3d_feature_2, image3d_feature_3, image3d_feature_4 = feature_list

        # the shape of the data is [16, 4, 3, 224, 224] ... description[batch, multi-view, channel, with, height]
        # n_samples is equal to the batch size of data loader
        n_samples, n_views, n_channels, h, w = image3d.shape
        if args.multigpu:
            image3d, graph, label = image3d.to(device), graph.to(device), label.to(device)
            image3d_feature_1, image3d_feature_2, image3d_feature_3, image3d_feature_4 \
                = [image3d_feature_1.to(device), image3d_feature_2.to(device), image3d_feature_3.to(device),
                   image3d_feature_4.to(device)]
        else:
            image3d, graph, label = image3d.to(device), graph.to(device), label.to(device)
            image3d_feature_1, image3d_feature_2, image3d_feature_3, image3d_feature_4 \
                = image3d_feature_1.to(device), image3d_feature_2.to(device), image3d_feature_3.to(
                device), image3d_feature_4.to(device)

        sample_num += image3d.shape[0]

        # forward
        image3d_encoder = image3d_encoder.to(device)
        graph_encoder = graph_encoder.to(device)

        if prompt is not None:
            prompt = prompt.to(device)

        criterion = criterion.to(device)

        linear_out_features = []
        if multi_scale:
            for layer in range(len(linear_predictor)):
                if layer == 0:
                    output_feature = linear_predictor[layer](image3d_feature_1)
                elif layer == 1:
                    output_feature = linear_predictor[layer](image3d_feature_2)
                elif layer == 2:
                    output_feature = linear_predictor[layer](image3d_feature_3)
                else:
                    output_feature = linear_predictor[layer](image3d_feature_4)
                output_feature = output_feature.to(device)
                linear_out_features.append(output_feature)

        feature_image3d = (image3d_encoder(image3d.reshape(n_samples * n_views, n_channels, h, w))
                           .reshape(n_samples, n_views, -1))
        feature_image3d_mean = feature_image3d.mean(1)

        node_rep, graph_rep = graph_encoder.forward_cl(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        loss_all = criterion(feature_image3d_mean, graph_rep)
        
        if prompt is not None:
            prompt_fusion_feature = prompt(linear_out_features)
            loss_prompt = criterion(prompt_fusion_feature, graph_rep)
            weight_loss_prompt = 0.25 * loss_prompt
        elif prompt is None and multi_scale:

            loss_layer1 = criterion(linear_out_features[0], graph_rep)
            loss_layer2 = criterion(linear_out_features[1], graph_rep)
            loss_layer3 = criterion(linear_out_features[2], graph_rep)
            loss_layer4 = criterion(linear_out_features[3], graph_rep)
            weight_loss_multi = 0.5 * loss_layer1 + 0.25 * loss_layer2 + 0.15 * loss_layer3 + 0.1 * loss_layer4
            # loss_multi = loss_layer1 + loss_layer2 + loss_layer3 + loss_layer4
            # loss = loss + 0.2 * loss_multi
        else:
            loss = loss_all
        if prompt is not None:
            loss = loss_all + weight_loss_prompt
        elif prompt is None and multi_scale:
            loss = loss_all + weight_loss_multi
        
        loss.backward()

        accu_loss += loss.detach()
        if prompt is not None:
            accu_prompt_loss += weight_loss_prompt.detach()
            data_loader.desc = ("[train epoch {}] total loss: {:.3f}; avg_loss: {:.3f}; prompt_loss: {:.3f}; ".
                                format(epoch, accu_loss.item(), accu_loss.item() / (step + 1),
                                       accu_prompt_loss.item() / (step + 1)))
        elif prompt is None and multi_scale:
            accu_multi_loss += weight_loss_multi.detach()
            data_loader.desc = ("[train epoch {}] total loss: {:.3f}; avg_loss: {:.3f}; multi_loss: {:.3f}; ".
                                format(epoch, accu_loss.item(), accu_loss.item() / (step + 1),
                                       accu_multi_loss.item() / (step + 1)))
        else:
            data_loader.desc = ("[train epoch {}] total loss: {:.3f}; avg_loss: {:.3f} ".
                                format(epoch, accu_loss.item(), accu_loss.item() / (step + 1)))
   
        optimizer.step()
        optimizer.zero_grad()

    train_dict = {
        'step': ((step + 1) + len(data_loader)) * epoch,
        'epoch': epoch + (step + 1) / len(data_loader),
        'total_loss': accu_loss.item(),
        'avg_loss': accu_loss.item() / (step + 1)
    }
    return train_dict
