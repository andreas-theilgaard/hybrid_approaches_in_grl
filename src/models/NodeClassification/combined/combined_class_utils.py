import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import torch.nn as nn
from src.models.model_utils import get_negative_samples
import numpy as np
from torch.utils.data import DataLoader


class SAGE(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout, apply_batchnorm: bool = False
    ):
        super(SAGE, self).__init__()

        self.apply_batchnorm = apply_batchnorm
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if self.apply_batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.apply_batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.apply_batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, apply_batchnorm):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.apply_batchnorm = apply_batchnorm

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        if self.apply_batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            if self.apply_batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.apply_batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        apply_batchnorm,
        att_heads=1,
        dataset=None,
    ):
        super(GAT, self).__init__()

        self.dropout = dropout
        self.apply_batchnorm = apply_batchnorm
        self.dataset = dataset

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=att_heads))
        if self.apply_batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * att_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * att_heads, hidden_channels, heads=att_heads))
            if self.apply_batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * att_heads))
        self.convs.append(GATConv(hidden_channels * att_heads, out_channels, concat=False, heads=1))

    def forward(self, x, adj_t):
        if self.dataset == "Cora":
            x = F.dropout(x, p=self.dropout, training=self.training)  # Cora
        for i, conv in enumerate(self.convs[:-1]):
            # x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, adj_t)
            if self.apply_batchnorm:
                x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class NodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        apply_batchnorm: bool = False,
    ):
        super(NodeClassifier, self).__init__()
        self.dropout = dropout
        self.apply_batchnorm = apply_batchnorm

        # Create linear and batchnorm layers
        self.linear = nn.ModuleList()
        self.linear.append(torch.nn.Linear(in_channels, hidden_channels))
        if apply_batchnorm:
            self.batch_norm = torch.nn.ModuleList()
            self.batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.linear.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if apply_batchnorm:
                self.batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))
        self.linear.append(torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        for i, lin_layer in enumerate(self.linear[:-1]):
            x = lin_layer(x)
            if self.apply_batchnorm:
                x = self.batch_norm[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear[-1](x)
        return x


def test_MLP(model, x, y, split_idx, evaluator, config):
    model.eval()
    with torch.no_grad():
        out = model(x)
        y_pred = out.argmax(dim=-1, keepdim=True)

        if config.dataset.dataset_name != "ogbn-mag":
            predictions = {
                "train": {"y_true": y[split_idx["train"]], "y_hat": y_pred[split_idx["train"]]},
                "val": {"y_true": y[split_idx["valid"]], "y_hat": y_pred[split_idx["valid"]]},
                "test": {"y_true": y[split_idx["test"]], "y_hat": y_pred[split_idx["test"]]},
            }
        else:
            y_true_train = y[split_idx["train"]["paper"]]
            y_true_valid = y[split_idx["valid"]["paper"]]
            y_true_test = y[split_idx["test"]["paper"]]
            predictions = {
                "train": {"y_true": y_true_train, "y_hat": y_pred[split_idx["train"]["paper"]]},
                "val": {"y_true": y_true_valid, "y_hat": y_pred[split_idx["valid"]["paper"]]},
                "test": {"y_true": y_true_test, "y_hat": y_pred[split_idx["test"]["paper"]]},
            }
        results = evaluator.collect_metrics(predictions)
        return results


def test_joint(MLP, deep, shallow, data_deep, split_idx, evaluator, config):
    MLP.eval()
    deep.eval()
    shallow.eval()
    with torch.no_grad():
        deep_out = deep(data_deep.x, data_deep.adj_t)
        shallow_out = shallow.weight
        combined_out = torch.cat([shallow_out, deep_out], dim=-1)
        out = MLP(combined_out)
        y_pred = out.argmax(dim=-1, keepdim=True)

        if config.dataset.dataset_name == "ogbn-mag":
            y_true_train = data_deep.y[split_idx["train"]["paper"]]
            y_true_valid = data_deep.y[split_idx["valid"]["paper"]]
            y_true_test = data_deep.y[split_idx["test"]["paper"]]
            predictions = {
                "train": {"y_true": y_true_train, "y_hat": y_pred[split_idx["train"]["paper"]]},
                "val": {"y_true": y_true_valid, "y_hat": y_pred[split_idx["valid"]["paper"]]},
                "test": {"y_true": y_true_test, "y_hat": y_pred[split_idx["test"]["paper"]]},
            }

        else:
            y_true_train = data_deep.y[split_idx["train"]]
            y_true_valid = data_deep.y[split_idx["valid"]]
            y_true_test = data_deep.y[split_idx["test"]]

            predictions = {
                "train": {"y_true": y_true_train, "y_hat": y_pred[split_idx["train"]]},
                "val": {"y_true": y_true_valid, "y_hat": y_pred[split_idx["valid"]]},
                "test": {"y_true": y_true_test, "y_hat": y_pred[split_idx["test"]]},
            }
        results = evaluator.collect_metrics(predictions)
        return results


def metrics(pred, label, type="accuracy"):
    if type == "accuracy":
        yhat = (pred > 0.5).float()
        return torch.mean((yhat == label).float()).item()


def full_train(model, data, criterion, optimizer, num_nodes):
    model.train()
    optimizer.zero_grad()

    # Positive edges
    pos_edge_index = data.edge_index
    pos_edge_index = pos_edge_index.to(data.x.device)
    pos_out = model(pos_edge_index[0], pos_edge_index[1])

    # Negative edges
    neg_edge_index = get_negative_samples(pos_edge_index, num_nodes, pos_edge_index.size(1))
    neg_edge_index.to(data.x.device)
    neg_out = model(neg_edge_index[0], neg_edge_index[1])

    # Combining positive and negative edges
    out = torch.cat([pos_out, neg_out], dim=0)
    y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0).to(data.x.device)
    acc = metrics(out, y, type="accuracy")
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item(), acc


def batch_train(model, data, criterion, optimizer, num_nodes, batch_size, config):
    model.train()

    # Positive edges
    pos_edge_index = data.edge_index
    pos_edge_index = pos_edge_index.to(config.device)

    loss_list = []
    acc_list = []
    for perm in DataLoader(range(pos_edge_index.size(1)), batch_size, shuffle=True):
        optimizer.zero_grad()

        edge = pos_edge_index[:, perm]
        pos_out = model(edge[0], edge[1])

        # Negative edges
        neg_edge_index = get_negative_samples(edge, num_nodes, edge.size(1))
        neg_edge_index = neg_edge_index.to(config.device)
        neg_out = model(neg_edge_index[0], neg_edge_index[1])
        assert pos_out.shape == neg_out.shape

        # Combining positive and negative edges
        out = torch.cat([pos_out, neg_out], dim=0)
        y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0).to(config.device)
        acc = metrics(out, y, type="accuracy")
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        acc_list.append(acc)
    return (np.mean(loss_list), np.mean(acc_list))
