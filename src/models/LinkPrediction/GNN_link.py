import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from tqdm import tqdm
from src.models.model_utils import set_seed, prepare_metric_cols
from src.models.model_utils import create_path
from src.data.data_utils import get_link_data_split
from src.models.model_utils import get_negative_samples
from torch_geometric.utils import to_undirected
from src.models.model_utils import get_k_laplacian_eigenvectors
from torch_geometric.utils import to_undirected
from src.models.metrics import METRICS



class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

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

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

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
        return torch.sigmoid(x)


# def train(config, model, predictor, data, split_edge, optimizer, batch_size):
#     model.train()
#     predictor.train()

#     pos_train_edge = split_edge["train"]["edge"].to(data.x.device)

#     total_loss = total_examples = 0
#     for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
#         optimizer.zero_grad()

#         h = model(data.x, data.adj_t)

#         edge = pos_train_edge[perm].t()

#         pos_out = predictor(h[edge[0]], h[edge[1]])
#         pos_loss = -torch.log(pos_out + 1e-15).mean()

#         edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long, device=h.device)
#         neg_out = predictor(h[edge[0]], h[edge[1]])
#         neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
#         loss = pos_loss + neg_loss
#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

#         optimizer.step()

#         num_examples = pos_out.size(0)
#         total_loss += loss.item() * num_examples
#         total_examples += num_examples

#     return total_loss / total_examples


def train(config, model, predictor, data, split_edge, optimizer, batch_size):
    criterion = torch.nn.BCELoss()
    model.train()
    predictor.train()

    pos_train_edge = split_edge["train"]["edge"].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = (
            model(data.x, data.adj_t)
            if config.dataset.dataset_name in ["ogbl-collab", "Twitch", "Flickr"]
            else model(data.x, data.edge_index)
        )

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        # pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        # edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long, device=h.device)
        edge = get_negative_samples(edge_index=edge, num_nodes=data.x.shape[0], num_neg_samples=edge.size(1))
        neg_out = predictor(h[edge[0]], h[edge[1]])
        predictions = torch.cat([pos_out, neg_out], dim=0)
        y = (
            torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
            .to(data.x.device)
            .unsqueeze(1)
        )
        loss = criterion(predictions, y)
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        # loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test_collab(config, model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)
    pos_train_edge = split_edge["train"]["edge"].to(h.device)

    pos_valid_edge = split_edge["valid"]["edge"].to(h.device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(h.device)
    pos_test_edge = split_edge["test"]["edge"].to(h.device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    predictions = {
        "train": {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred},
        "val": {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred},
        "test": {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred},
    }
    results = evaluator.collect_metrics(predictions)
    return results


@torch.no_grad()
def evaluate_data(type_, model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.edge_index)

    pos_test_edge = split_edge[type_]["edge"].to(h.device)
    if type_ != "train":
        neg_test_edge = split_edge[type_]["edge_neg"].to(h.device)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    if type_ != "train":
        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

    return (pos_test_pred, neg_test_pred) if type_ != "train" else (pos_test_pred, None)


def test(config, model, predictor, data, split_edge, evaluator, batch_size):
    if config.dataset.dataset_name in ["ogbl-collab", "Twitch", "Flickr"]:
        return test_collab(
            config=config,
            model=model,
            predictor=predictor,
            data=data,
            split_edge=split_edge,
            evaluator=evaluator,
            batch_size=batch_size,
        )
    else:
        pos_train_pred, _ = evaluate_data(
            type_="train",
            model=model,
            predictor=predictor,
            data=data["train"],
            split_edge=split_edge,
            evaluator=evaluator,
            batch_size=batch_size,
        )
        pos_valid_pred, neg_valid_pred = evaluate_data(
            type_="valid",
            model=model,
            predictor=predictor,
            data=data["valid"],
            split_edge=split_edge,
            evaluator=evaluator,
            batch_size=batch_size,
        )
        pos_test_pred, neg_test_pred = evaluate_data(
            type_="test",
            model=model,
            predictor=predictor,
            data=data["test"],
            split_edge=split_edge,
            evaluator=evaluator,
            batch_size=batch_size,
        )

        predictions = {
            "train": {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred},
            "val": {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred},
            "test": {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred},
        }
        results = evaluator.collect_metrics(predictions)
        return results


def GNN_link_trainer(dataset, config, training_args, save_path, log=None, Logger=None, seeds=None):
    data = dataset[0]
    edge_index = data.edge_index

    if "edge_weight" in data:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)

    if config.dataset.dataset_name in ["ogbl-collab", "ogbl-citation2"]:
        data = T.ToSparseTensor()(data)
        split_edge = dataset.get_edge_split()

    elif config.dataset.dataset_name in ["Cora", "Flickr", "CiteSeer", "Twitch", "PubMed"]:
        if data.is_directed():
            data.edge_index = to_undirected(data.edge_index)
        data = T.ToSparseTensor()(data)

        train_data, val_data, test_data = get_link_data_split(data, dataset_name=config.dataset.dataset_name)
        split_edge = {
            "train": {"edge": train_data.pos_edge_label_index.T},
            "valid": {
                "edge": val_data.pos_edge_label_index.T,
                "edge_neg": val_data.neg_edge_label_index.T,
            },
            "test": {
                "edge": test_data.pos_edge_label_index.T,
                "edge_neg": test_data.neg_edge_label_index.T,
            },
        }

    data = data.to(config.device)
    evaluator = METRICS(
        metrics_list=config.dataset.metrics, task=config.dataset.task, dataset=config.dataset.dataset_name
    )

    if config.dataset.GNN.extra_info:
        embedding = torch.load(config.dataset[config.model_type].extra_info, map_location=config.device)
        X = torch.cat([data.x, embedding], dim=-1)
        data.x = X

    if config.dataset[config.model_type].use_spectral:
        X = get_k_laplacian_eigenvectors(
            data=(split_edge["train"]["edge"]).T,
            dataset=dataset,
            k=config.dataset[config.model_type].K,
            is_undirected=True,
            num_nodes=data.x.shape[0],
            for_link=True,
            edge_split=split_edge,
        )
        data.x = torch.cat([data.x, X], dim=-1)

    data = data.to(config.device)

    for seed in seeds:
        set_seed(seed)
        Logger.start_run()

        if config.dataset.GNN.model == "GraphSage":
            model = SAGE(
                in_channels=data.x.shape[1],
                hidden_channels=training_args.hidden_channels,
                num_layers=training_args.num_layers,
                dropout=training_args.dropout,
                out_channels=training_args.hidden_channels,
            ).to(config.device)
        elif config.dataset.GNN.model == "GCN":
            model = GCN(
                in_channels=data.x.shape[1],
                hidden_channels=training_args.hidden_channels,
                num_layers=training_args.num_layers,
                dropout=training_args.dropout,
                out_channels=training_args.hidden_channels,
            ).to(config.device)
        elif config.dataset.GNN.model == "GAT":
            model = GAT(
                in_channels=data.x.shape[1],
                hidden_channels=training_args.hidden_channels,
                num_layers=training_args.num_layers,
                dropout=training_args.dropout,
                out_channels=training_args.hidden_channels,
                att_heads=1,
            ).to(config.device)
        else:
            raise ValueError(f"Model {config.dataset.GNN.model} not implemented")

        predictor = LinkPredictor(
            in_channels=training_args.hidden_channels,
            hidden_channels=training_args.hidden_channels,
            out_channels=1,
            num_layers=training_args.num_layers,
            dropout=training_args.dropout,
        ).to(config.device)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=training_args.lr,
            weight_decay=training_args.weight_decay if training_args.weight_decay else 0,
        )

        prog_bar = tqdm(range(training_args.epochs))

        for i, epoch in enumerate(prog_bar):
            loss = (
                train(
                    config=config,
                    model=model,
                    predictor=predictor,
                    data=data, 
                    split_edge=split_edge,
                    optimizer=optimizer,
                    batch_size=training_args.batch_size,
                )
                if config.dataset.dataset_name in ["ogbl-collab", "Flickr", "Twitch"]
                else train(
                    config=config,
                    model=model,
                    predictor=predictor,
                    data=train_data.to(config.device),
                    split_edge=split_edge,
                    optimizer=optimizer,
                    batch_size=training_args.batch_size,
                )
            )

            results = (
                test(
                    config=config,
                    model=model,
                    predictor=predictor,
                    data=data,
                    split_edge=split_edge,
                    evaluator=evaluator,
                    batch_size=training_args.batch_size,
                )
                if config.dataset.dataset_name in ["ogbl-collab", "Flickr", "Twitch"]
                else test(
                    config=config,
                    model=model,
                    predictor=predictor,
                    data={"train": train_data, "valid": val_data, "test": test_data},
                    split_edge=split_edge,
                    evaluator=evaluator,
                    batch_size=training_args.batch_size,
                )
            )

            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    f"Train {config.dataset.track_metric}": results["train"][config.dataset.track_metric],
                    f"Val {config.dataset.track_metric}": results["val"][config.dataset.track_metric],
                    f"Test {config.dataset.track_metric}": results["test"][config.dataset.track_metric],
                }
            )

            Logger.add_to_run(loss=loss, results=results)

        Logger.end_run()
        model_save_path = save_path + f"/model_{seed}.pth"
        log.info(f"saved model at {model_save_path}")
        torch.save(model.state_dict(), model_save_path)
        if "save_to_folder" in config:
            create_path(config.save_to_folder)
            additional_save_path = f"{config.save_to_folder}/{config.dataset.task}/{config.dataset.dataset_name}/{config.dataset.DIM}/{config.model_type}"
            create_path(f"{additional_save_path}")
            create_path(f"{additional_save_path}/models")
            used_emb = (
                (config.dataset.GNN.extra_info.split("/"))[-1] if config.dataset.GNN.extra_info else False
            )
            MODEL_PATH = (
                f"{additional_save_path}/models/{config.dataset.GNN.model}_{used_emb}_model_{seed}.pth"
            )
            torch.save(model.state_dict(), MODEL_PATH)

    if "save_to_folder" in config:
        Logger.save_results(
            additional_save_path
            + f"/results_{config.dataset.GNN.model}_{used_emb}_{config.dataset.GNN.use_spectral}.json"
        )

    Logger.save_results(save_path + "/results.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
    Logger.save_value(
        {
            "loss": loss,
            f"Test {config.dataset.track_metric}": results["test"][config.dataset.track_metric],
        }
    )
