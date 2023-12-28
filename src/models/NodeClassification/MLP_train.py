from src.models.NodeClassification.models import MLP_model

import torch
from src.models.model_utils import set_seed
from src.models.model_utils import create_path
from src.models.model_utils import prepare_metric_cols
from src.models.model_utils import get_k_laplacian_eigenvectors
from torch_geometric.utils import to_undirected
from src.models.metrics import METRICS
from torch_geometric.data import Data


def mlp_node_classification(dataset, config, training_args, log, save_path, seeds, Logger):
    """
    Function that instanties and runs a MLP model for the node classification task

    args:
        dataset:
            torch geometric dataset
        config:
            config form yaml file
        training_args:
            traning arguments from config, used for shorten the reference to these args
        save_path:
            path to the current hydra folder
        seeds:
            list of seeds that will be used for the current experiment
        Logger:
            the Logger class as defined in src/models/logger.py

    """
    data = dataset[0]
    if "ogb" in config.dataset.dataset_name:
        split_idx = dataset.get_idx_split()
    else:
        split_idx = {"train": data.train_mask, "valid": data.val_mask, "test": data.test_mask}

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
            data=data,
            dataset=dataset,
            k=config.dataset[config.model_type].K,
            is_undirected=True,
            for_link=False,
            edge_split=split_idx,
            num_nodes=data.x.shape[0],
        )
    if config.dataset[config.model_type].random:
        x = torch.normal(0, 1, (data.x.shape[0], 128))

    X = x.to(config.device)
    y = data.y.to(config.device)
    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    evaluator = METRICS(
        metrics_list=config.dataset.metrics, task=config.dataset.task, dataset=config.dataset.dataset_name
    )

    for seed in seeds:
        set_seed(seed=seed)
        Logger.start_run()

        model = MLP_model(
            device=config.device,
            in_channels=X.shape[-1],
            hidden_channels=training_args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=training_args.num_layers,
            dropout=training_args.dropout,
            log=log,
            logger=Logger,
            apply_batchnorm=training_args.batchnorm,
            config=config,
        )

        model.fit(
            X=X,
            y=y,
            epochs=training_args.epochs,
            split_idx=split_idx,
            evaluator=evaluator,
            lr=training_args.lr,
            weight_decay=training_args.weight_decay if training_args.weight_decay else 0,
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
