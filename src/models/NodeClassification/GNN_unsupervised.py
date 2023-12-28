import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.models.model_utils import set_seed, prepare_metric_cols
from src.models.metrics import METRICS
from src.models.model_utils import create_path
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric.utils import to_undirected, negative_sampling
from src.models.model_utils import get_laplacian
from torch_geometric.utils import to_undirected


def decoder(decode_type, beta, z_i, z_j):
    if decode_type == "dot":
        return beta + ((z_i * z_j).sum(dim=-1))
    elif decode_type == "dist":
        return beta - torch.norm(z_i - z_j, dim=-1)
    else:
        raise ValueError("Decoder method not yet implemented")


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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, att_heads=8):
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


class GNN:
    def __init__(self, GNN_type, in_channels, hidden_channels, out_channels, dropout, num_layers):
        self.GNN_type = GNN_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_layers = num_layers

    def get_gnn_model(self):
        if self.GNN_type == "GCN":
            model = GCN(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )

        elif self.GNN_type == "GraphSage":
            model = SAGE(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )

        elif self.GNN_type == "GAT":
            model = GAT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
                att_heads=1,
            )
        return model

    def get_negative_samples(self, edge_index, num_nodes, num_neg_samples):
        return negative_sampling(edge_index, num_neg_samples=num_neg_samples, num_nodes=num_nodes)

    def metrics(self, pred, label, type="accuracy"):
        if type == "accuracy":
            yhat = (pred > 0.5).float()
            return torch.mean((yhat == label).float()).item()

    def full_train(self, model, data, criterion, optimizer, num_nodes, beta, training_args, config):
        model.train()
        optimizer.zero_grad()

        Z = model(data.x, data.edge_index)

        # Positive edges
        pos_edge_index = data.edge_index
        pos_edge_index = pos_edge_index.to(config.device)
        pos_out = decoder(
            decode_type=training_args.decode_type,
            beta=beta,
            z_i=Z[pos_edge_index[0]],
            z_j=Z[pos_edge_index[1]],
        )

        # Negative edges
        neg_edge_index = self.get_negative_samples(pos_edge_index, num_nodes, pos_edge_index.size(1))
        neg_edge_index = neg_edge_index.to(config.device)
        neg_out = decoder(
            decode_type=training_args.decode_type,
            beta=beta,
            z_i=Z[neg_edge_index[0]],
            z_j=Z[neg_edge_index[1]],
        )

        # Combining positive and negative edges
        out = torch.cat([pos_out, neg_out], dim=0)
        y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0).to(config.device)
        acc = self.metrics(out, y, type="accuracy")
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        return loss.item(), acc

    def batch_train(
        self, model, data, criterion, optimizer, batch_size, num_nodes, beta, training_args, config
    ):
        model.train()
        # Positive edges
        pos_edge_index = data.edge_index
        pos_edge_index = pos_edge_index.to(config.device)
        loss_list = []
        acc_list = []
        for perm in DataLoader(range(pos_edge_index.size(1)), batch_size, shuffle=True):
            optimizer.zero_grad()

            Z = model(data.x, data.edge_index)

            edge = pos_edge_index[:, perm]
            pos_out = decoder(
                decode_type=training_args.decode_type, beta=beta, z_i=Z[edge[0]], z_j=Z[edge[1]]
            )

            # Negative edges
            edge = self.get_negative_samples(edge, num_nodes, edge.size(1))
            edge = edge.to(config.device)
            neg_out = decoder(
                decode_type=training_args.decode_type, beta=beta, z_i=Z[edge[0]], z_j=Z[edge[1]]
            )
            assert pos_out.shape == neg_out.shape

            # Combining positive and negative edges
            out = torch.cat([pos_out, neg_out], dim=0)
            y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0).to(
                config.device
            )
            acc = self.metrics(out, y, type="accuracy")
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            acc_list.append(acc)
        return (np.mean(loss_list), np.mean(acc_list))

    def train(
        self,
        train_batch,
        model,
        data,
        criterion,
        optimizer,
        batch_size,
        num_nodes,
        beta,
        training_args,
        config,
    ):
        if train_batch:
            loss, acc = self.batch_train(
                model=model,
                data=data,
                criterion=criterion,
                optimizer=optimizer,
                batch_size=batch_size,
                num_nodes=num_nodes,
                beta=beta,
                training_args=training_args,
                config=config,
            )
        else:
            loss, acc = self.full_train(
                model=model,
                data=data,
                criterion=criterion,
                optimizer=optimizer,
                num_nodes=num_nodes,
                beta=beta,
                training_args=training_args,
                config=config,
            )
        return loss, acc


def GNN_unsupervised_trainer(dataset, config, training_args, log, save_path, seeds, Logger):
    beta = 0
    data = dataset[0]
    if config.dataset.dataset_name in ["ogbn-arxiv", "ogbn-mag"]:
        if config.dataset.dataset_name == "ogbn-mag":
            data = Data(
                x=data.x_dict["paper"],
                edge_index=data.edge_index_dict[("paper", "cites", "paper")],
                y=data.y_dict["paper"],
            )
        if data.is_directed():
            data.edge_index = to_undirected(data.edge_index)
        data.edge_index = to_undirected(data.edge_index)
    else:
        if data.is_directed():
            data.edge_index = to_undirected(data.edge_index)

    data = data.to(config.device)

    if config.dataset.GNN_DIRECT.extra_info:
        embedding = torch.load(config.dataset[config.model_type].extra_info, map_location=config.device)
        X = torch.cat([data.x, embedding], dim=-1)
        data.x = X

    if config.dataset[config.model_type].use_spectral:
        X = get_laplacian(
            K=config.dataset[config.model_type].K,
            task="NodeClassification",
            dataset_name=config.dataset.dataset_name,
            config=config,
        )
        data.x = torch.cat([data.x, X], dim=-1)

    data = data.to(config.device)

    num_nodes = data.x.shape[0]

    for seed in seeds:
        set_seed(seed)
        Logger.start_run()
        GNN_object = GNN(
            GNN_type=config.dataset[config.model_type].model,
            in_channels=data.x.shape[-1],
            hidden_channels=training_args.hidden_channels,
            out_channels=training_args.hidden_channels,
            dropout=training_args.dropout,
            num_layers=training_args.num_layers,
        )
        model = GNN_object.get_gnn_model()
        if (
            config.dataset.dataset_name in ["ogbn-products", "ogbn-mag"]
            and config.dataset[config.model_type].model == "GCN"
        ):
            # Pre-compute GCN normalization.
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            data.adj_t = adj_t

        data = data.to(config.device)
        model = model.to(config.device)

        if training_args.decode_type in ["dist", "dot"]:
            beta = torch.nn.Parameter(torch.tensor(training_args.init_beta))
            optimizer = torch.optim.Adam(
                list(model.parameters()) + [beta],
                lr=training_args.lr,
                weight_decay=training_args.weight_decay if training_args.weight_decay else 0,
            )
        else:
            optimizer = torch.optim.Adam(
                list(model.parameters()),
                lr=training_args.lr,
                weight_decay=training_args.weight_decay if training_args.weight_decay else 0,
            )
        criterion = torch.nn.BCEWithLogitsLoss()
        prog_bar = tqdm(range(training_args.epochs))
        for i, epoch in enumerate(prog_bar):
            loss, acc = GNN_object.train(
                model=model,
                train_batch=training_args.train_batch,
                data=data,
                criterion=criterion,
                optimizer=optimizer,
                batch_size=training_args.batch_size,
                num_nodes=num_nodes,
                beta=beta,
                training_args=training_args,
                config=config,
            )
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    f"Train acc": acc,
                }
            )
        Logger.end_run()
        model.train()

        model_save_path = save_path + f"/model_{seed}.pth"
        log.info(f"saved model at {model_save_path}")
        torch.save(model.state_dict(), model_save_path)
        if "save_to_folder" in config:
            model.eval()
            with torch.no_grad():
                embeddings = model(data.x, data.edge_index)
            create_path(config.save_to_folder)
            additional_save_path = f"{config.save_to_folder}/{config.dataset.task}/{config.dataset.dataset_name}/{config.dataset.DIM}/{config.model_type}"
            create_path(f"{additional_save_path}")
            create_path(f"{additional_save_path}/models")
            used_emb = (
                (config.dataset.GNN_DIRECT.extra_info.split("/"))[-1]
                if config.dataset.GNN_DIRECT.extra_info
                else False
            )
            MODEL_PATH = f"{additional_save_path}/models/GNN_DIRECT_{config.dataset.GNN_DIRECT.model}_{used_emb}_model_{seed}_{config.dataset.GNN_DIRECT.use_spectral}.pth"
            torch.save(embeddings, MODEL_PATH)
