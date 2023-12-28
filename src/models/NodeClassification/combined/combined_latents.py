import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from src.models.model_utils import set_seed
from src.models.LinkPrediction.combined.combined_link_utils import ModelWeights
from src.models.shallow import ShallowModel, initialize_embeddings
from src.models.metrics import METRICS
from src.models.logger import LoggerClass
from src.models.model_utils import prepare_metric_cols
import copy
from src.models.NodeClassification.combined.combined_class_utils import (
    NodeClassifier,
    SAGE,
    full_train,
    batch_train,
    test_joint,
    GCN,
    GAT,
)
from torch_geometric.data import Data
from src.models.model_utils import create_path


def create_and_fit_shallow(data, dataset, split_idx, training_args, config):
    init_embeddings = initialize_embeddings(
        num_nodes=data.x.shape[0],
        data=data,
        dataset=dataset,
        method=training_args.init,
        dim=training_args.embedding_dim,
        for_link=False,
        edge_split=split_idx,
    )
    init_embeddings = init_embeddings.to(config.device)
    shallow = ShallowModel(
        num_nodes=data.x.shape[0],
        embedding_dim=training_args.embedding_dim,
        beta=training_args.init_beta,
        init_embeddings=init_embeddings,
        decoder_type=training_args.decode_type,
        device=config.device,
    ).to(config.device)
    optimizer = torch.optim.Adam(list(shallow.parameters()), lr=training_args.shallow_lr)
    prog_bar = tqdm(range(training_args.SHALLOW_WARM_START))
    criterion = nn.BCEWithLogitsLoss()
    for epoch in prog_bar:
        if training_args.SHALLOW_TRAIN_BATCH:
            loss, acc = batch_train(
                model=shallow,
                data=data,
                criterion=criterion,
                optimizer=optimizer,
                num_nodes=data.x.shape[0],
                batch_size=training_args.SHALLOW_WARM_BATCH_SIZE,
                config=config,
            )
        else:
            loss, acc = full_train(
                model=shallow,
                data=data,
                criterion=criterion,
                optimizer=optimizer,
                num_nodes=data.x.shape[0],
            )
        prog_bar.set_postfix({"loss": loss, "Train Acc.": acc})
    return shallow


def fit_combined_latents(config, dataset, training_args, Logger, log, seeds, save_path):
    data = dataset[0]
    directed = True
    if config.dataset.dataset_name == "ogbn-mag":
        data = Data(
            x=data.x_dict["paper"],
            edge_index=data.edge_index_dict[("paper", "cites", "paper")],
            y=data.y_dict["paper"],
        )
    if data.is_directed():
        directed = False
        data.edge_index = to_undirected(data.edge_index)

    if config.dataset.dataset_name in ["ogbn-arxiv", "ogbn-mag"]:
        split_idx = dataset.get_idx_split()
    else:
        split_idx = {"train": data.train_mask, "valid": data.val_mask, "test": data.test_mask}

    data = data.to(config.device)
    data_deep = copy.deepcopy(data)
    data_deep = T.ToSparseTensor()(data_deep)
    if not directed:
        data_deep.adj_t = data_deep.adj_t.to_symmetric()
    data_deep = data_deep.to(config.device)

    for counter, seed in enumerate(seeds):
        set_seed(seed)
        Logger.start_run()

        shallow = create_and_fit_shallow(
            data=data, dataset=dataset, split_idx=split_idx, training_args=training_args, config=config
        )
        shallow_embeddings = (shallow.embeddings).to(config.device)
        # torch.save(shallow_embeddings.weight.data.cpu(),'emb.pth')
        # create gnn
        # shallow_embeddings = torch.load('emb.pth', map_location=config.device)

        if training_args.deep_model == "GraphSage":
            deep = SAGE(
                in_channels=data.num_features,
                hidden_channels=training_args.deep_hidden_channels,
                out_channels=training_args.deep_out_dim,
                num_layers=training_args.deep_num_layers,
                dropout=training_args.deep_dropout,
                apply_batchnorm=training_args.APPLY_BATCHNORM,
            ).to(config.device)
        elif training_args.deep_model == "GCN":
            deep = GCN(
                in_channels=data.num_features,
                hidden_channels=training_args.deep_hidden_channels,
                out_channels=training_args.deep_out_dim,
                num_layers=training_args.deep_num_layers,
                dropout=training_args.deep_dropout,
                apply_batchnorm=training_args.APPLY_BATCHNORM,
            ).to(config.device)
        elif training_args.deep_model == "GAT":
            deep = GAT(
                in_channels=data.num_features,
                hidden_channels=training_args.deep_hidden_channels,
                out_channels=training_args.deep_out_dim,
                num_layers=training_args.deep_num_layers,
                dropout=training_args.deep_dropout,
                apply_batchnorm=training_args.APPLY_BATCHNORM,
                dataset=config.dataset.dataset_name,
            ).to(config.device)

        # create mlp
        MLP = NodeClassifier(
            in_channels=shallow.embeddings.weight.data.shape[1] + training_args.deep_out_dim,
            hidden_channels=training_args.MLP_HIDDEN,
            out_channels=dataset.num_classes,
            dropout=training_args.MLP_DROPOUT,
            apply_batchnorm=training_args.APPLY_BATCHNORM,
            num_layers=training_args.MLP_NUM_LAYERS,
        ).to(config.device)

        params_combined = [
            {"params": list(deep.parameters()), "lr": training_args.deep_lr},
            {"params": list(MLP.parameters()), "lr": training_args.MLP_LR},
            # {"params": shallow_embeddings.parameters(), "lr": training_args.shallow_lr},
        ]

        optimizer = torch.optim.Adam(params_combined)
        prog_bar = tqdm(range(training_args.epochs))
        train_idx = (
            split_idx["train"].to(config.device)
            if config.dataset.dataset_name != "ogbn-mag"
            else split_idx["train"]["paper"].to(config.device)
        )
        if train_idx.dtype == torch.bool:
            train_idx = torch.where(train_idx.float() == 1)[0]
        criterion = torch.nn.CrossEntropyLoss()
        evaluator = METRICS(metrics_list=config.dataset.metrics, task="NodeClassification", dataset=dataset)
        # control_model_weights = ModelWeights(
        #     direction=training_args.direction,
        #     shallow_frozen_epochs=training_args.shallow_frozen_epochs,
        #     deep_frozen_epochs=training_args.deep_frozen_epochs,
        # )
        if len(data_deep.y.shape) == 1:
            data_deep.y = data_deep.y.unsqueeze(1)

        for epoch in prog_bar:
            deep.train()
            MLP.train()
            # shallow_embeddings.train()

            # control_model_weights.step(epoch=epoch, shallow=shallow_embeddings, deep=deep)
            optimizer.zero_grad()

            deep_out = deep(data_deep.x, data_deep.adj_t)[train_idx]
            shallow_out = shallow_embeddings(train_idx)
            combined_out = torch.cat([shallow_out, deep_out], dim=-1).to(config.device)
            out = MLP(combined_out)
            loss = criterion(
                out, (data_deep.y[train_idx]).type(torch.LongTensor).squeeze(1).to(config.device)
            )
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(MLP.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(shallow_embeddings.parameters(), 1.0)
            optimizer.step()

            # evaluate
            results = test_joint(
                deep=deep,
                MLP=MLP,
                shallow=shallow_embeddings,
                data_deep=data_deep,
                split_idx=split_idx,
                evaluator=evaluator,
                config=config,
            )
            prog_bar.set_postfix(
                {
                    "loss": loss.item(),
                    f"Train {config.dataset.track_metric}": results["train"][config.dataset.track_metric],
                    f"Val {config.dataset.track_metric}": results["val"][config.dataset.track_metric],
                    f"Test {config.dataset.track_metric}": results["test"][config.dataset.track_metric],
                }
            )

            Logger.add_to_run(loss=loss.item(), results=results)

        Logger.end_run()
    Logger.save_results(save_path + "/class_combined_latents.json")
    if "save_to_folder" in config:
        create_path(config.save_to_folder)
        additional_save_path = f"{config.save_to_folder}/{config.dataset.task}/{config.dataset.dataset_name}/{config.dataset.DIM}/{config.model_type}"
        create_path(f"{additional_save_path}")
        Logger.save_results(additional_save_path + f"/class_combined_latents_{training_args.deep_model}.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
