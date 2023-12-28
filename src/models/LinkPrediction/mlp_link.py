import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.models.model_utils import set_seed
from src.models.model_utils import prepare_metric_cols
from src.models.model_utils import get_k_laplacian_eigenvectors
from torch_geometric.utils import to_undirected
from src.models.metrics import METRICS
from src.models.model_utils import create_path
from src.data.data_utils import get_link_data_split


class LinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super(LinkPredictor, self).__init__()
        self.dropout = dropout

        # Create linear and batchnorm layers
        self.linear = nn.ModuleList()
        self.linear.append(torch.nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.linear.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.linear.append(torch.nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.linear:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for i, lin_layer in enumerate(self.linear[:-1]):
            x = lin_layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear[-1](x)
        return torch.sigmoid(x)


class MLP_LinkPrediction:
    def __init__(
        self,
        device,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        log,
        save_path,
        logger,
        config,
    ):
        self.device = device
        self.model = LinkPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.model.to(device)
        self.log = log
        self.save_path = save_path
        self.logger = logger
        self.config = config

    def train(self, X, split_edge, optimizer, batch_size):
        self.model.train()

        positive_edges = split_edge["train"]["edge"].to(self.device)
        total_loss, total_examples = 0, 0
        loss_fn = nn.BCELoss()

        for perm in DataLoader(range(positive_edges.size(0)), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            pos_edge = positive_edges[perm].t()

            # option 1
            positive_preds = self.model(X[pos_edge[0]], X[pos_edge[1]])
            pos_loss = -torch.log(positive_preds + 1e-15).mean()
            neg_edge = torch.randint(0, X.size(0), pos_edge.size(), dtype=torch.long, device=self.device)
            neg_preds = self.model(X[neg_edge[0]], X[neg_edge[1]])
            neg_loss = -torch.log(1 - neg_preds + 1e-15).mean()
            loss = pos_loss + neg_loss

            loss.backward()
            optimizer.step()

            num_examples = positive_preds.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        return total_loss / total_examples

    @torch.no_grad()
    def test(self, x, split_edge, evaluator, batch_size):
        self.model.eval()

        pos_train_edge = split_edge["train"]["edge"].to(self.device)
        if self.config.dataset.dataset_name in ["ogbl-vessel"]:
            neg_train_edge = split_edge["train"]["edge_neg"].to(self.device)

        pos_valid_edge = split_edge["valid"]["edge"].to(self.device)
        neg_valid_edge = split_edge["valid"]["edge_neg"].to(self.device)
        pos_test_edge = split_edge["test"]["edge"].to(self.device)
        neg_test_edge = split_edge["test"]["edge_neg"].to(self.device)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
            edge = pos_train_edge[perm].t()
            pos_train_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
            edge = pos_valid_edge[perm].t()
            pos_valid_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
            edge = neg_valid_edge[perm].t()
            neg_valid_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            pos_test_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [self.model(x[edge[0]], x[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        predictions = {
                "train": {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred},
                "val": {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred},
                "test": {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred},
            }
        results = evaluator.collect_metrics(predictions)
        return results

    @torch.no_grad()
    def test_citation(self, x, split_edge, evaluator, batch_size):
        self.model.eval()

        def test_split(split):
            source = split_edge[split]["source_node"].to(x.device)
            target = split_edge[split]["target_node"].to(x.device)
            target_neg = split_edge[split]["target_node_neg"].to(x.device)

            pos_preds = []
            for perm in DataLoader(range(source.size(0)), batch_size):
                src, dst = source[perm], target[perm]
                pos_preds += [self.model(x[src], x[dst]).squeeze().cpu()]
            pos_pred = torch.cat(pos_preds, dim=0)

            neg_preds = []
            source = source.view(-1, 1).repeat(1, 1000).view(-1)
            target_neg = target_neg.view(-1)
            for perm in DataLoader(range(source.size(0)), batch_size):
                src, dst_neg = source[perm], target_neg[perm]
                neg_preds += [self.model(x[src], x[dst_neg]).squeeze().cpu()]
            neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

            return pos_pred, neg_pred

        train = test_split("eval_train")
        valid = test_split("valid")
        test = test_split("test")

        predictions = {
            "train": {"y_pred_pos": train[0], "y_pred_neg": train[1]},
            "val": {"y_pred_pos": valid[0], "y_pred_neg": valid[1]},
            "test": {"y_pred_pos": test[0], "y_pred_neg": test[1]},
        }
        results = evaluator.collect_metrics(predictions)

        return results

    def fit(self, X, split_edge, lr, batch_size, epochs, evaluator):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        prog_bar = tqdm(range(epochs))

        for i, epoch in enumerate(prog_bar):
            loss = self.train(X, split_edge, optimizer, batch_size)

            results = self.test(X, split_edge=split_edge, evaluator=evaluator, batch_size=batch_size)
            
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    f"Train {self.config.dataset.track_metric}": results["train"][
                        self.config.dataset.track_metric
                    ],
                    f"Val {self.config.dataset.track_metric}": results["val"][
                        self.config.dataset.track_metric
                    ],
                    f"Test {self.config.dataset.track_metric}": results["test"][
                        self.config.dataset.track_metric
                    ],
                }
            )

            self.logger.add_to_run(loss=loss, results=results)

        self.logger.save_value(
            {
                "loss": loss,
                f"Test {self.config.dataset.track_metric}": results["test"][self.config.dataset.track_metric],
            }
        )


def mlp_LinkPrediction(dataset, config, training_args, log, save_path, seeds, Logger):
    data = dataset[0]
    if config.dataset.dataset_name in ["ogbl-collab", "ogbl-vessel", "ogbl-citation2"]:
        split_edge = dataset.get_edge_split()
    elif config.dataset.dataset_name in ["Cora", "Flickr", "CiteSeer", "PubMed", "Twitch"]:
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

    if (
        config.dataset[config.model_type].saved_embeddings
        and config.dataset[config.model_type].using_features
    ):
        embedding = torch.load(config.dataset[config.model_type].saved_embeddings, map_location=config.device)
        x = torch.cat([data.x, embedding], dim=-1)
    if (
        config.dataset[config.model_type].saved_embeddings
        and not config.dataset[config.model_type].using_features
    ):
        embedding = torch.load(config.dataset[config.model_type].saved_embeddings, map_location=config.device)
        x = embedding
    if (
        not config.dataset[config.model_type].saved_embeddings
        and config.dataset[config.model_type].using_features
    ):
        x = data.x
    if config.dataset[config.model_type].use_spectral:
        if data.is_directed():
            data.edge_index = to_undirected(data.edge_index)
        x = get_k_laplacian_eigenvectors(
            data=(split_edge["train"]["edge"]).T,
            dataset=dataset,
            k=config.dataset[config.model_type].K,
            is_undirected=True,
            num_nodes=data.x.shape[0],
            for_link=True,
            edge_split=split_edge,
        )

    if config.dataset[config.model_type].random:
        x = torch.normal(0, 1, (data.x.shape[0], 128))

    X = x.to(config.device)
    evaluator = METRICS(
        metrics_list=config.dataset.metrics, task=config.dataset.task, dataset=config.dataset.dataset_name
    )

    for seed in seeds:
        set_seed(seed=seed)
        Logger.start_run()

        model = MLP_LinkPrediction(
            device=config.device,
            in_channels=X.shape[-1],
            hidden_channels=training_args.hidden_channels,
            out_channels=1,
            num_layers=training_args.num_layers,
            dropout=training_args.dropout,
            log=log,
            save_path=save_path,
            logger=Logger,
            config=config,
        )
        model.fit(
            X=X,
            split_edge=split_edge,
            lr=training_args.lr,
            batch_size=training_args.batch_size,
            epochs=training_args.epochs,
            evaluator=evaluator,
        )
        Logger.end_run()

    Logger.save_results(save_path + "/results.json")
    if "save_to_folder" in config:
        create_path(config.save_to_folder)
        additional_save_path = f"{config.save_to_folder}/{config.dataset.task}/{config.dataset.dataset_name}/{config.dataset.DIM}/{config.model_type}"
        create_path(f"{additional_save_path}")
        saved_embeddings_path = (
            False
            if not config.dataset.DownStream.saved_embeddings
            else (config.dataset.DownStream.saved_embeddings.split("/"))[-1]
        )
        print(saved_embeddings_path)
        Logger.save_results(
            additional_save_path
            + f"/results_{saved_embeddings_path}_{config.dataset.DownStream.using_features}_{config.dataset.DownStream.use_spectral}_{config.dataset.DownStream.random}.json"
        )
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
