import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.nn import GATConv,GCNConv,SAGEConv

class NodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        apply_batchnorm: bool,
    ):
        super(NodeClassifier, self).__init__()
        self.dropout = dropout
        self.apply_batchnorm = apply_batchnorm

        # Create linear and batchnorm layers
        self.linear = nn.ModuleList()
        self.linear.append(torch.nn.Linear(in_channels, hidden_channels))
        if self.apply_batchnorm:
            self.batch_norm = torch.nn.ModuleList()
            self.batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.linear.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.apply_batchnorm:
                self.batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))
        self.linear.append(torch.nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.linear:
            lin.reset_parameters()
        for bn in self.batch_norm:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin_layer in enumerate(self.linear[:-1]):
            x = lin_layer(x)
            if self.apply_batchnorm:
                x = self.batch_norm[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear[-1](x)
        return torch.log_softmax(x, dim=-1)


class MLP_model:
    def __init__(
        self,
        device,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        apply_batchnorm: bool,
        log,
        logger,
        config,
    ):
        self.device = device
        self.model = NodeClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            apply_batchnorm=apply_batchnorm,
        )
        self.model.to(device)
        self.log = log
        self.logger = logger
        self.config = config

    def train(self, X, y, train_idx, optimizer):
        self.model.train()
        optimizer.zero_grad()
        out = self.model(X[train_idx])
        loss = F.nll_loss(out, y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(self, X, y, split_idx, evaluator):
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
            y_hat = out.argmax(dim=-1, keepdim=True)
            predictions = {
                    "train": {"y_true": y[split_idx["train"]], "y_hat": y_hat[split_idx["train"]]},
                    "val": {"y_true": y[split_idx["valid"]], "y_hat": y_hat[split_idx["valid"]]},
                    "test": {"y_true": y[split_idx["test"]], "y_hat": y_hat[split_idx["test"]]},
                }
            results = evaluator.collect_metrics(predictions)
            return results

    def fit(self, X, y, epochs: int, split_idx, evaluator, lr, weight_decay):
        train_idx = split_idx["train"].to(self.device)

        prog_bar = tqdm(range(epochs))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        for i, epoch in enumerate(prog_bar):
            loss = self.train(X=X, y=y, train_idx=train_idx, optimizer=optimizer)
            result = self.test(X=X, y=y, split_idx=split_idx, evaluator=evaluator)
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    f"Train {self.config.dataset.track_metric}": result["train"][
                        self.config.dataset.track_metric
                    ],
                    f"Val {self.config.dataset.track_metric}": result["val"][
                        self.config.dataset.track_metric
                    ],
                    f"Test {self.config.dataset.track_metric}": result["test"][
                        self.config.dataset.track_metric
                    ],
                }
            )

            self.logger.add_to_run(loss=loss, results=result)

        self.logger.save_value(
            {
                "loss": loss,
                f"Test {self.config.dataset.track_metric}": result["test"][self.config.dataset.track_metric],
            }
        )

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
        return x.log_softmax(dim=-1)


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

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.apply_batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


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

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.apply_batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)