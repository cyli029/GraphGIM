import torch
import torch.nn.functional as F
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.utils import add_self_loops


class GINConv(MessagePassing):
    """
        Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr='add'):
        super(GINConv, self).__init__(aggr=aggr)
        # multi-layer perception
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embeddings = self.bond_encoder(edge_attr)
        # update using the former node feature
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        out = self.mlp((1 + self.eps) * x + aggr_out)
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layers, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        super(GNN, self).__init__()
        self.num_layer = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim=emb_dim)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layers):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr='add'))

        # List of batch norms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise Exception(f"{self.JK} is undefined.")
        return node_representation

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.load_state_dict(torch.load(model_file))


class GINGraphPre(torch.nn.Module):
    """
        Extension of GIN to incorporate edge information by concatenation.

        Args:
            num_layer (int): the number of GNN layers
            emb_dim (int): dimensionality of embeddings
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            graph_pooling (str): sum, mean, max, attention, set2set
            gnn_type: gin, gcn, graphsage, gat

        See https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layers, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin"):
        super(GINGraphPre, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.JK = JK

        if self.num_layers < 2:
            raise ValueError("Number of GIN layers must be greater than 1.")

        self.gnn = GNN(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == 'max':
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim,1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling[:-1] == 'set2set':
            self.multi = 2
        else:
            self.multi = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.multi * (self.num_layers + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.multi * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        model_state_dict = torch.load(model_file)['graph']
        self.gnn.load_state_dict(model_state_dict)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch)), node_representation


class GINGraphRepr(torch.nn.Module):
    def __init__(self, num_tasks=512,
                 num_layers=5,
                 emb_dim=300,
                 residual=False,
                 drop_ratio=0,
                 JK="last",
                 graph_pooling="mean"):
        """

        GIN Graph Pooling Module
                Args:
                    num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表征的维度，dimension of graph representation).
                    num_layers (int, optional): number of GINConv layers. Defaults to 5.
                    emb_dim (int, optional): dimension of node embedding. Defaults to 300.
                    residual (bool, optional): adding residual connection or not. Defaults to False.
                    drop_ratio (float, optional): dropout rate. Defaults to 0.
                    JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
                    graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "sum".

                Out:
                    graph representation
        """
        super(GINGraphRepr, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.gnn_node = GNN(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio)

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling[:-1] == 'set2set':
            self.multi = 2
        else:
            self.multi = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.multi * (self.num_layers + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.multi * self.emb_dim, self.num_tasks)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn_node(x, edge_index, edge_attr)
        graph_representation = self.graph_pred_linear(self.pool(node_representation, batch))
        return self.normalize(graph_representation)

    def normalize(self, graph_representation):
        """
        Normalize the graph representation features.
        Args:
            graph_representation (Tensor): The graph representation tensor to be normalized.
        Returns:
            Tensor: The normalized graph representation.
        """
        return F.normalize(graph_representation, p=2, dim=1)
