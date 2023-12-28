import torch
from src.models.NodeClassification.models import GCN,SAGE,GAT
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from src.models.model_utils import set_seed, prepare_metric_cols
from src.models.metrics import METRICS
from src.models.model_utils import create_path
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from src.models.get_laplacian import get_laplacian


class GNN:
    def __init__(
        self,
        GNN_type,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        apply_batchnorm,
        dataset,
    ):
        self.GNN_type = GNN_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.apply_batchnorm = apply_batchnorm
        self.dataset = dataset

    def get_gnn_model(self):
        if self.GNN_type == "GCN":
            model = GCN(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
                apply_batchnorm=self.apply_batchnorm,
            )

        elif self.GNN_type == "GraphSage":
            model = SAGE(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
                apply_batchnorm=self.apply_batchnorm,
            )

        elif self.GNN_type == "GAT":
            model = GAT(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_layers,
                dropout=self.dropout,
                apply_batchnorm=self.apply_batchnorm,
                att_heads=1,
                dataset=self.dataset,
            )

        return model

    def train(self, model, data, train_idx, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])

        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def test(self, model, data, split_idx, evaluator, config):
        model.eval()

        out = model(data.x, data.adj_t)  # data.edge_index
        y_pred = out.argmax(dim=-1, keepdim=True)

        y_true_train = data.y[split_idx["train"]]
        y_true_valid = data.y[split_idx["valid"]]
        y_true_test = data.y[split_idx["test"]]

        predictions = {
            "train": {"y_true": y_true_train, "y_hat": y_pred[split_idx["train"]]},
            "val": {"y_true": y_true_valid, "y_hat": y_pred[split_idx["valid"]]},
            "test": {"y_true": y_true_test, "y_hat": y_pred[split_idx["test"]]},
        }
        results = evaluator.collect_metrics(predictions)
        return results


def GNN_trainer(dataset, config, training_args, log, save_path, seeds, Logger):
    data = dataset[0]
    evaluator = METRICS(metrics_list=config.dataset.metrics, task=config.dataset.task, dataset=config.dataset.dataset_name)

    if config.dataset.dataset_name in ["ogbn-arxiv"]:
        data.adj_t = data.adj_t.to_symmetric()
    else:
        if data.is_directed():
            data = T.ToSparseTensor()(data)
            data.adj_t = data.adj_t.to_symmetric()
        else:
            data = T.ToSparseTensor()(data)

    data = data.to(config.device)

    if config.dataset.GNN.extra_info:
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
    if config.dataset.dataset_name in ["ogbn-arxiv"]:
        split_idx = dataset.get_idx_split()
    else:
        split_idx = {"train": data.train_mask, "valid": data.val_mask, "test": data.test_mask}

    train_idx = split_idx["train"].to(config.device)

    if len(data.y.shape) == 1:
        data.y = data.y.unsqueeze(1)

    for seed in seeds:
        set_seed(seed)
        Logger.start_run()
        GNN_object = GNN(
            GNN_type=config.dataset[config.model_type].model,
            in_channels=data.x.shape[-1],
            hidden_channels=training_args.hidden_channels,
            out_channels=dataset.num_classes,
            dropout=training_args.dropout,
            num_layers=training_args.num_layers,
            apply_batchnorm=training_args.batchnorm,
            dataset=config.dataset.dataset_name,
        )
        model = GNN_object.get_gnn_model()
        data = data.to(config.device)
        model = model.to(config.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_args.lr,
            weight_decay=training_args.weight_decay if training_args.weight_decay else 0,
        )
        prog_bar = tqdm(range(training_args.epochs))

        for i, epoch in enumerate(prog_bar):
            loss = GNN_object.train(model, data, train_idx, optimizer)
            result = GNN_object.test(model, data, split_idx, evaluator, config)
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    f"Train {config.dataset.track_metric}": result["train"][config.dataset.track_metric],
                    f"Val {config.dataset.track_metric}": result["val"][config.dataset.track_metric],
                    f"Test {config.dataset.track_metric}": result["test"][config.dataset.track_metric],
                }
            )
            Logger.add_to_run(loss=loss, results=result)

        Logger.end_run()
        model.train()

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
            MODEL_PATH = f"{additional_save_path}/models/{config.dataset.GNN.model}_{used_emb}_model_{seed}_{config.dataset.GNN.use_spectral}.pth"
            torch.save(model.state_dict(), MODEL_PATH)

    if "save_to_folder" in config:
        Logger.save_results(
            additional_save_path
            + f"/results_{config.dataset.GNN.model}_{used_emb}_{config.dataset.GNN.use_spectral}.json"
        )

    Logger.save_value(
        {"loss": loss, f"Test {config.dataset.track_metric}": result["test"][config.dataset.track_metric]}
    )
    Logger.save_results(save_path + "/results.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
