import torch
import torch.nn as nn
from src.data.get_data import DataLoader as DL
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.data_utils import get_link_data_split
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor


def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = True


class ModelWeights:
    def __init__(self, direction: str, shallow_frozen_epochs: int, deep_frozen_epochs: int) -> None:
        self.direction = direction
        self.shallow_frozen_epochs = shallow_frozen_epochs
        self.deep_frozen_epochs = deep_frozen_epochs
        self.shallow_is_frozen = None
        self.deep_is_frozen = None

    def step(self, epoch, shallow=None, deep=None, predictor=False):
        shallow_freeze, deep_freeze = self.update_freeze_states(epoch)
        if shallow_freeze != self.shallow_is_frozen:
            if shallow_freeze:
                freeze_model_params(shallow)
            else:
                unfreeze_model_params(shallow)
        if deep_freeze != self.deep_is_frozen:
            if deep_freeze:
                freeze_model_params(deep)
                if not isinstance(predictor, bool):
                    freeze_model_params(predictor)
            else:
                unfreeze_model_params(deep)
                if not isinstance(predictor, bool):
                    unfreeze_model_params(predictor)

        self.deep_is_frozen = deep_freeze
        self.shallow_is_frozen = shallow_freeze

    def update_freeze_states(self, epoch):
        if self.direction == "shallow_first":
            if epoch < self.shallow_frozen_epochs:
                shallow_freeze = False
                deep_freeze = True
            elif (
                self.shallow_frozen_epochs <= epoch
                and epoch < self.shallow_frozen_epochs + self.deep_frozen_epochs
            ):
                shallow_freeze = True
                deep_freeze = False
            else:
                shallow_freeze = False
                deep_freeze = False
        elif self.direction == "deep_first":
            if epoch < self.deep_frozen_epochs:
                shallow_freeze = True
                deep_freeze = False
            elif (
                self.deep_frozen_epochs <= epoch
                and epoch < self.shallow_frozen_epochs + self.deep_frozen_epochs
            ):
                shallow_freeze = False
                deep_freeze = True
            else:
                shallow_freeze = False
                deep_freeze = False
        return shallow_freeze, deep_freeze


def decode(W, node_i, node_j, gamma, type_):
    if type_ == "dot":
        return gamma + ((W[node_i] * W[node_j]).sum(dim=-1))
    elif type_ == "dist":
        return gamma - torch.norm(W[node_i, :] - W[node_j, :], dim=-1)


def predict(shallow_logit, lambda_, deep_logit, training_args):
    if training_args.balance:
        return (1 - lambda_) * shallow_logit + lambda_ * deep_logit
    else:
        return shallow_logit + lambda_ * deep_logit


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE_Direct(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout, gamma=0.0, decode_type=None
    ):
        super(SAGE_Direct, self).__init__()
        self.decode_type = decode_type
        if decode_type == "dist":
            self.gamma = nn.Parameter(torch.tensor(gamma))
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, adj_t, node_i, node_j):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.decode_type == "dist":
            return self.gamma - torch.norm(x[node_i, :] - x[node_j, :], dim=-1)
        return (x[node_i, :] * x[node_j, :]).sum(dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, att_heads=1):
        super(GAT, self).__init__()

        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=att_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * att_heads, hidden_channels, heads=att_heads))
        self.convs.append(GATConv(hidden_channels * att_heads, out_channels, concat=False, heads=1))

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GCN_Direct(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout, gamma=0.0, decode_type=None
    ):
        super(GCN, self).__init__()
        self.decode_type = decode_type
        if decode_type == "dist":
            self.gamma = nn.Parameter(torch.tensor(gamma))

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def forward(self, x, adj_t, node_i, node_j):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.decode_type == "dist":
            return self.gamma - torch.norm(x[node_i, :] - x[node_j, :], dim=-1)
        return (x[node_i, :] * x[node_j, :]).sum(dim=-1)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def test_joint(
    shallow,
    deep,
    split_edge,
    x,
    adj_t,
    λ,
    evaluator,
    batch_size=65536,
    direct=False,
    gamma=None,
    training_args=None,
):
    shallow.eval()
    deep.eval()

    with torch.no_grad():
        # get training edges
        pos_train_edge = split_edge["train"]["edge"].to(x.device)

        # get validation edges
        pos_valid_edge = split_edge["valid"]["edge"].to(x.device)
        neg_valid_edge = split_edge["valid"]["edge_neg"].to(x.device)

        # get test edges
        pos_test_edge = split_edge["test"]["edge"].to(x.device)
        neg_test_edge = split_edge["test"]["edge_neg"].to(x.device)

        if not direct:
            W = deep(x, adj_t)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
            edge = pos_train_edge[perm].t()
            # compute shallow
            shallow_out = shallow(edge[0], edge[1])
            # compute deep
            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )
            # make prediction
            pos_train_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out,
                        lambda_=λ,
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
            edge = pos_valid_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )
            pos_valid_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out,
                        lambda_=λ,
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
            edge = neg_valid_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )
            neg_valid_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out,
                        lambda_=λ,
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )
            pos_test_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out,
                        lambda_=λ,
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )
            neg_test_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out,
                        lambda_=λ,
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        predictions = {
            "train": {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred},
            "val": {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred},
            "test": {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred},
        }
        results = evaluator.collect_metrics(predictions)
        return results


def test_joint_with_predictor(
    shallow, deep, predictor, split_edge, x, adj_t, λ, evaluator, batch_size=65536, training_args=None
):
    shallow.eval()
    deep.eval()
    predictor.eval()

    with torch.no_grad():
        # get training edges
        pos_train_edge = split_edge["train"]["edge"].to(x.device)

        # get validation edges
        pos_valid_edge = split_edge["valid"]["edge"].to(x.device)
        neg_valid_edge = split_edge["valid"]["edge_neg"].to(x.device)

        # get test edges
        pos_test_edge = split_edge["test"]["edge"].to(x.device)
        neg_test_edge = split_edge["test"]["edge_neg"].to(x.device)

        W = deep(x, adj_t)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
            edge = pos_train_edge[perm].t()
            # compute shallow
            shallow_out = shallow(edge[0], edge[1])
            # compute deep
            deep_o = predictor(W[edge[0]], W[edge[1]]).squeeze().cpu()
            # make prediction
            pos_train_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out.cpu(),
                        lambda_=λ.cpu(),
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
            edge = pos_valid_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            deep_o = predictor(W[edge[0]], W[edge[1]]).squeeze().cpu()
            pos_valid_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out.cpu(),
                        lambda_=λ.cpu(),
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
            edge = neg_valid_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            deep_o = predictor(W[edge[0]], W[edge[1]]).squeeze().cpu()
            neg_valid_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out.cpu(),
                        lambda_=λ.cpu(),
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            deep_o = predictor(W[edge[0]], W[edge[1]]).squeeze().cpu()
            pos_test_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out.cpu(),
                        lambda_=λ.cpu(),
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            deep_o = predictor(W[edge[0]], W[edge[1]]).squeeze().cpu()
            neg_test_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out.cpu(),
                        lambda_=λ.cpu(),
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        predictions = {
            "train": {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred},
            "val": {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred},
            "test": {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred},
        }
        results = evaluator.collect_metrics(predictions)
        return results


def test_joint_with_emb_combined(shallow, deep, predictor, split_edge, x, adj_t, evaluator, batch_size=65536):
    shallow.eval()
    deep.eval()
    predictor.eval()

    with torch.no_grad():
        # get training edges
        pos_train_edge = split_edge["train"]["edge"].to(x.device)

        # get validation edges
        pos_valid_edge = split_edge["valid"]["edge"].to(x.device)
        neg_valid_edge = split_edge["valid"]["edge_neg"].to(x.device)

        # get test edges
        pos_test_edge = split_edge["test"]["edge"].to(x.device)
        neg_test_edge = split_edge["test"]["edge_neg"].to(x.device)

        W = deep(x, adj_t)
        Z = shallow.weight.data.to(x.device)
        concat_embeddings = torch.cat([Z, W], dim=-1)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
            edge = pos_train_edge[perm].t()
            deep_o = predictor(concat_embeddings[edge[0]], concat_embeddings[edge[1]]).squeeze().cpu()
            pos_train_preds += [torch.sigmoid(deep_o)]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
            edge = pos_valid_edge[perm].t()
            deep_o = predictor(concat_embeddings[edge[0]], concat_embeddings[edge[1]]).squeeze().cpu()
            pos_valid_preds += [torch.sigmoid(deep_o)]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
            edge = neg_valid_edge[perm].t()
            deep_o = predictor(concat_embeddings[edge[0]], concat_embeddings[edge[1]]).squeeze().cpu()
            neg_valid_preds += [torch.sigmoid(deep_o)]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            deep_o = predictor(concat_embeddings[edge[0]], concat_embeddings[edge[1]]).squeeze().cpu()
            pos_test_preds += [torch.sigmoid(deep_o)]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            deep_o = predictor(concat_embeddings[edge[0]], concat_embeddings[edge[1]]).squeeze().cpu()
            neg_test_preds += [torch.sigmoid(deep_o)]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        predictions = {
            "train": {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred},
            "val": {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred},
            "test": {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred},
        }
        results = evaluator.collect_metrics(predictions)
        return results


def test_indi(
    shallow,
    deep,
    split_edge,
    x,
    adj_t,
    evaluator,
    batch_size=65536,
    direct=False,
    gamma=None,
    training_args=None,
):
    shallow.eval()
    deep.eval()

    with torch.no_grad():
        # get training edges
        pos_train_edge = split_edge["train"]["edge"].to(x.device)

        # get validation edges
        pos_valid_edge = split_edge["valid"]["edge"].to(x.device)
        neg_valid_edge = split_edge["valid"]["edge_neg"].to(x.device)

        # get test edges
        pos_test_edge = split_edge["test"]["edge"].to(x.device)
        neg_test_edge = split_edge["test"]["edge_neg"].to(x.device)

        if not direct:
            W = deep(x, adj_t)

        pos_train_preds_shallow = []
        pos_train_preds_deep = []
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
            edge = pos_train_edge[perm].t()

            shallow_out = shallow(edge[0], edge[1])
            pos_train_preds_shallow += [torch.sigmoid(shallow_out)]

            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )
            pos_train_preds_deep += [torch.sigmoid(deep_o)]

        pos_train_pred_shallow = torch.cat(pos_train_preds_shallow, dim=0)
        pos_train_pred_deep = torch.cat(pos_train_preds_deep, dim=0)

        pos_valid_preds_shallow = []
        pos_valid_preds_deep = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
            edge = pos_valid_edge[perm].t()

            shallow_out = shallow(edge[0], edge[1])
            pos_valid_preds_shallow += [torch.sigmoid(shallow_out)]

            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )
            pos_valid_preds_deep += [torch.sigmoid(deep_o)]

        pos_valid_pred_shallow = torch.cat(pos_valid_preds_shallow, dim=0)
        pos_valid_pred_deep = torch.cat(pos_valid_preds_deep, dim=0)

        neg_valid_preds_shallow = []
        neg_valid_preds_deep = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
            edge = neg_valid_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])

            neg_valid_preds_shallow += [torch.sigmoid(shallow_out)]
            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )

            neg_valid_preds_deep += [torch.sigmoid(deep_o)]

        neg_valid_pred_shallow = torch.cat(neg_valid_preds_shallow, dim=0)
        neg_valid_pred_deep = torch.cat(neg_valid_preds_deep, dim=0)

        pos_test_preds_shallow = []
        pos_test_preds_deep = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()

            shallow_out = shallow(edge[0], edge[1])
            pos_test_preds_shallow += [torch.sigmoid(shallow_out)]

            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )
            pos_test_preds_deep += [torch.sigmoid(deep_o)]

        pos_test_pred_shallow = torch.cat(pos_test_preds_shallow, dim=0)
        pos_test_pred_deep = torch.cat(pos_test_preds_deep, dim=0)

        neg_test_preds_shallow = []
        neg_test_preds_deep = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()

            shallow_out = shallow(edge[0], edge[1])
            neg_test_preds_shallow += [torch.sigmoid(shallow_out)]

            deep_o = (
                deep(x, adj_t, edge[0], edge[1])
                if direct
                else decode(W, edge[0], edge[1], gamma, type_=training_args.deep_decode)
            )
            neg_test_preds_deep += [torch.sigmoid(deep_o)]

        neg_test_pred_shallow = torch.cat(neg_test_preds_shallow, dim=0)
        neg_test_pred_deep = torch.cat(neg_test_preds_deep, dim=0)

        predictions_shallow = {
            "train": {"y_pred_pos": pos_train_pred_shallow, "y_pred_neg": neg_valid_pred_shallow},
            "val": {"y_pred_pos": pos_valid_pred_shallow, "y_pred_neg": neg_valid_pred_shallow},
            "test": {"y_pred_pos": pos_test_pred_shallow, "y_pred_neg": neg_test_pred_shallow},
        }
        results_shallow = evaluator.collect_metrics(predictions_shallow)

        predictions_deep = {
            "train": {"y_pred_pos": pos_train_pred_deep, "y_pred_neg": neg_valid_pred_deep},
            "val": {"y_pred_pos": pos_valid_pred_deep, "y_pred_neg": neg_valid_pred_deep},
            "test": {"y_pred_pos": pos_test_pred_deep, "y_pred_neg": neg_test_pred_deep},
        }
        results_deep = evaluator.collect_metrics(predictions_deep)

    return (results_shallow, results_deep)


def test_indi_with_predictor(
    shallow, deep, split_edge, x, adj_t, evaluator, batch_size=65536, linkpredictor=False
):
    shallow.eval()
    deep.eval()
    linkpredictor.eval()

    with torch.no_grad():
        # get training edges
        pos_train_edge = split_edge["train"]["edge"].to(x.device)

        # get validation edges
        pos_valid_edge = split_edge["valid"]["edge"].to(x.device)
        neg_valid_edge = split_edge["valid"]["edge_neg"].to(x.device)

        # get test edges
        pos_test_edge = split_edge["test"]["edge"].to(x.device)
        neg_test_edge = split_edge["test"]["edge_neg"].to(x.device)

        W = deep(x, adj_t)

        pos_train_preds_shallow = []
        pos_train_preds_deep = []
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
            edge = pos_train_edge[perm].t()

            shallow_out = shallow(edge[0], edge[1])
            pos_train_preds_shallow += [torch.sigmoid(shallow_out)]
            pos_train_preds_deep += [torch.sigmoid(linkpredictor(W[edge[0]], W[edge[1]]).squeeze().cpu())]
        pos_train_pred_shallow = torch.cat(pos_train_preds_shallow, dim=0)
        pos_train_pred_deep = torch.cat(pos_train_preds_deep, dim=0)

        pos_valid_preds_shallow = []
        pos_valid_preds_deep = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
            edge = pos_valid_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            pos_valid_preds_shallow += [torch.sigmoid(shallow_out)]
            pos_valid_preds_deep += [torch.sigmoid(linkpredictor(W[edge[0]], W[edge[1]]).squeeze().cpu())]
        pos_valid_pred_shallow = torch.cat(pos_valid_preds_shallow, dim=0)
        pos_valid_pred_deep = torch.cat(pos_valid_preds_deep, dim=0)

        neg_valid_preds_shallow = []
        neg_valid_preds_deep = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
            edge = neg_valid_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            neg_valid_preds_shallow += [torch.sigmoid(shallow_out)]
            neg_valid_preds_deep += [torch.sigmoid(linkpredictor(W[edge[0]], W[edge[1]]).squeeze().cpu())]
        neg_valid_pred_shallow = torch.cat(neg_valid_preds_shallow, dim=0)
        neg_valid_pred_deep = torch.cat(neg_valid_preds_deep, dim=0)

        pos_test_preds_shallow = []
        pos_test_preds_deep = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            pos_test_preds_shallow += [torch.sigmoid(shallow_out)]
            pos_test_preds_deep += [torch.sigmoid(linkpredictor(W[edge[0]], W[edge[1]]).squeeze().cpu())]
        pos_test_pred_shallow = torch.cat(pos_test_preds_shallow, dim=0)
        pos_test_pred_deep = torch.cat(pos_test_preds_deep, dim=0)

        neg_test_preds_shallow = []
        neg_test_preds_deep = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            shallow_out = shallow(edge[0], edge[1])
            neg_test_preds_shallow += [torch.sigmoid(shallow_out)]
            neg_test_preds_deep += [torch.sigmoid(linkpredictor(W[edge[0]], W[edge[1]]).squeeze().cpu())]
        neg_test_pred_shallow = torch.cat(neg_test_preds_shallow, dim=0)
        neg_test_pred_deep = torch.cat(neg_test_preds_deep, dim=0)

        predictions_shallow = {
            "train": {"y_pred_pos": pos_train_pred_shallow, "y_pred_neg": neg_valid_pred_shallow},
            "val": {"y_pred_pos": pos_valid_pred_shallow, "y_pred_neg": neg_valid_pred_shallow},
            "test": {"y_pred_pos": pos_test_pred_shallow, "y_pred_neg": neg_test_pred_shallow},
        }
        results_shallow = evaluator.collect_metrics(predictions_shallow)

        predictions_deep = {
            "train": {"y_pred_pos": pos_train_pred_deep, "y_pred_neg": neg_valid_pred_deep},
            "val": {"y_pred_pos": pos_valid_pred_deep, "y_pred_neg": neg_valid_pred_deep},
            "test": {"y_pred_pos": pos_test_pred_deep, "y_pred_neg": neg_test_pred_deep},
        }
        results_deep = evaluator.collect_metrics(predictions_deep)

    return (results_shallow, results_deep)


def get_split_edge(data, dataset, config, training_args):
    data_splits = None
    if config.dataset.dataset_name in ["ogbl-collab", "ogbl-citation2"]:
        split_edge = dataset.get_edge_split()

    elif config.dataset.dataset_name in ["Cora", "Flickr", "CiteSeer", "Twitch", "PubMed"]:
        train_data, val_data, test_data = get_link_data_split(data, dataset_name=config.dataset.dataset_name)
        edge_weight_in = data.edge_weight if "edge_weight" in data else None
        edge_weight_in = edge_weight_in.float() if edge_weight_in else edge_weight_in
        split_edge = {
            "train": {"edge": train_data.pos_edge_label_index.T, "weight": edge_weight_in},
            "valid": {
                "edge": val_data.pos_edge_label_index.T,
                "edge_neg": val_data.neg_edge_label_index.T,
            },
            "test": {
                "edge": test_data.pos_edge_label_index.T,
                "edge_neg": test_data.neg_edge_label_index.T,
            },
        }
        data_splits = {"train": train_data, "val": val_data, "test": test_data}

    data = (split_edge["train"]["edge"]).T

    return (data, split_edge, data_splits)
