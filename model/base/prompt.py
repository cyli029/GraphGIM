import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def attention(query, key, value, mask, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.w_q = nn.Linear(self.hidden_size, 32)
        self.w_k = nn.Linear(self.hidden_size, 32)
        self.w_v = nn.Linear(self.hidden_size, 32)

        self.dense = nn.Linear(32, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, feature_hidden, init_hidden):
        query = self.w_q(feature_hidden)
        key = self.w_k(feature_hidden)
        value = self.w_v(feature_hidden)

        padding_mask = (init_hidden != 0) + 0.0
        mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
        x, attn = attention(query, key, value, mask)

        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


'''
       self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
       self.alpha.data.fill_(0.1)
       self.norm = nn.LayerNorm(args.hidden_size)
'''


class prompt_generator(nn.Module):
    def __init__(self, args):
        super(prompt_generator, self).__init__()
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.cls = nn.Parameter(torch.randn(1, self.embedding_size), requires_grad=True)
        self.linear = nn.Linear(self.embedding_size, self.hidden_size)
        self.attention_layer_1 = AttentionLayer(args)
        self.attention_layer_2 = AttentionLayer(args)

    def forward(self, feature_map):
        device = feature_map[0].device
        batch_size = feature_map[0].size()[0]
        multi_feature_nums = len(feature_map)
        for i in range(multi_feature_nums):
            feature_map[i] = feature_map[i].unsqueeze(dim=1)

        feature_map = torch.cat(feature_map, dim=1)
        x0 = torch.zeros(batch_size, self.embedding_size).unsqueeze(dim=1).to(device)
        concat_x0_feature = torch.cat((x0, feature_map), dim=1)

        index = torch.zeros(batch_size, 1, self.embedding_size, dtype=torch.long).to(device)
        cls_expanded = self.cls.expand(batch_size, -1, -1).to(device)
        concat_x0_feature = concat_x0_feature.scatter_(1, index, cls_expanded)
        init_hidden = concat_x0_feature
        hidden_states = self.attention_layer_1(concat_x0_feature, init_hidden)
        hidden_states = self.attention_layer_2(hidden_states, init_hidden)

        cls_hidden = hidden_states[:, 0]
        cls_hidden = self.linear(cls_hidden)
        return cls_hidden

