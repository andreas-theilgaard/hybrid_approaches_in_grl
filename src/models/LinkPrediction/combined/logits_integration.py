import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.model_utils import set_seed, get_negative_samples
from src.models.LinkPrediction.combined.combined_link_utils import (
    LinkPredictor,
    SAGE,
    get_split_edge,
    ModelWeights,
    GCN,
    predict,
    GAT,
)
from src.models.shallow import ShallowModel, initialize_embeddings
from src.models.metrics import METRICS
from src.models.model_utils import prepare_metric_cols
import numpy as np
from src.models.model_utils import create_path


def test_joint_with_predictor_collab(
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


@torch.no_grad()
def evaluate_data_joint(type_, shallow, deep, predictor, deep_data, batch_size, split_edge, training_args, λ):
    shallow.eval()
    deep.eval()
    predictor.eval()

    h = deep(deep_data.x, deep_data.edge_index)

    pos_test_edge = split_edge[type_]["edge"].to(h.device)
    if type_ != "train":
        neg_test_edge = split_edge[type_]["edge_neg"].to(h.device)

    pos_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        # shallow
        shallow_out = shallow(edge[0], edge[1])
        deep_o = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
        pos_preds += [
            torch.sigmoid(
                predict(
                    shallow_logit=shallow_out.cpu(),
                    lambda_=λ.cpu(),
                    deep_logit=deep_o,
                    training_args=training_args,
                )
            )
        ]

    pos_pred = torch.cat(pos_preds, dim=0)

    if type_ != "train":
        neg_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            # shallow
            shallow_out = shallow(edge[0], edge[1])
            deep_o = predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()
            neg_preds += [
                torch.sigmoid(
                    predict(
                        shallow_logit=shallow_out.cpu(),
                        lambda_=λ.cpu(),
                        deep_logit=deep_o,
                        training_args=training_args,
                    )
                )
            ]

        neg_pred = torch.cat(neg_preds, dim=0)

    out = {"pos": pos_pred, "neg": neg_pred if type_ != "train" else None}
    return out


def test_joint_with_predictor(
    shallow,
    deep,
    split_edge,
    x,
    adj_t,
    evaluator,
    batch_size=65536,
    linkpredictor=False,
    config=None,
    data_splits=None,
    λ=None,
    training_args=None,
):
    if config.dataset.dataset_name in ["ogbl-collab", "Flickr", "Twitch"]:
        return test_joint_with_predictor_collab(
            shallow=shallow,
            deep=deep,
            split_edge=split_edge,
            x=x,
            adj_t=adj_t,
            evaluator=evaluator,
            batch_size=batch_size,
            predictor=linkpredictor,
            λ=λ,
            training_args=training_args,
        )
    else:
        train_out = evaluate_data_joint(
            type_="train",
            shallow=shallow,
            deep=deep,
            predictor=linkpredictor,
            batch_size=batch_size,
            split_edge=split_edge,
            deep_data=data_splits["train"],
            λ=λ,
            training_args=training_args,
        )
        val_out = evaluate_data_joint(
            type_="valid",
            shallow=shallow,
            deep=deep,
            predictor=linkpredictor,
            batch_size=batch_size,
            split_edge=split_edge,
            deep_data=data_splits["val"],
            λ=λ,
            training_args=training_args,
        )
        test_out = evaluate_data_joint(
            type_="test",
            shallow=shallow,
            deep=deep,
            predictor=linkpredictor,
            batch_size=batch_size,
            split_edge=split_edge,
            deep_data=data_splits["test"],
            λ=λ,
            training_args=training_args,
        )
        predictions = {
            "train": {"y_pred_pos": train_out["pos"], "y_pred_neg": val_out["neg"]},
            "val": {"y_pred_pos": val_out["pos"], "y_pred_neg": val_out["neg"]},
            "test": {"y_pred_pos": test_out["pos"], "y_pred_neg": test_out["neg"]},
        }
        results = evaluator.collect_metrics(predictions)

        return results


@torch.no_grad()
def evaluate_data(type_, shallow, deep, predictor, deep_data, batch_size, split_edge):
    shallow.eval()
    deep.eval()
    predictor.eval()

    h = deep(deep_data.x, deep_data.edge_index)

    pos_test_edge = split_edge[type_]["edge"].to(h.device)
    if type_ != "train":
        neg_test_edge = split_edge[type_]["edge_neg"].to(h.device)

    pos_preds_shallow = []
    pos_preds_deep = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        # shallow
        shallow_out = shallow(edge[0], edge[1])
        pos_preds_shallow += [torch.sigmoid(shallow_out)]
        # deep
        pos_preds_deep += [torch.sigmoid(predictor(h[edge[0]], h[edge[1]]).squeeze().cpu())]

    pos_pred_shallow = torch.cat(pos_preds_shallow, dim=0)
    pos_pred_deep = torch.cat(pos_preds_deep, dim=0)

    if type_ != "train":
        neg_preds_shallow = []
        neg_preds_deep = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            # shallow
            shallow_out = shallow(edge[0], edge[1])
            neg_preds_shallow += [torch.sigmoid(shallow_out)]
            # deep
            neg_preds_deep += [torch.sigmoid(predictor(h[edge[0]], h[edge[1]]).squeeze().cpu())]

        neg_pred_shallow = torch.cat(neg_preds_shallow, dim=0)
        neg_pred_deep = torch.cat(neg_preds_deep, dim=0)

    out = {
        "pos_shallow": pos_pred_shallow,
        "pos_deep": pos_pred_deep,
        "neg_shallow": neg_pred_shallow if type_ != "train" else None,
        "neg_deep": neg_pred_deep if type_ != "train" else None,
    }
    return out


def test_indi_with_predictor_collab(
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


def test_indi_with_predictor(
    shallow,
    deep,
    split_edge,
    x,
    adj_t,
    evaluator,
    batch_size=65536,
    linkpredictor=False,
    config=None,
    data_splits=None,
):
    if config.dataset.dataset_name in ["ogbl-collab", "Flickr", "Twitch"]:
        return test_indi_with_predictor_collab(
            shallow=shallow,
            deep=deep,
            split_edge=split_edge,
            x=x,
            adj_t=adj_t,
            evaluator=evaluator,
            batch_size=batch_size,
            linkpredictor=linkpredictor,
        )
    else:
        train_out = evaluate_data(
            type_="train",
            shallow=shallow,
            deep=deep,
            predictor=linkpredictor,
            batch_size=batch_size,
            split_edge=split_edge,
            deep_data=data_splits["train"],
        )
        val_out = evaluate_data(
            type_="valid",
            shallow=shallow,
            deep=deep,
            predictor=linkpredictor,
            batch_size=batch_size,
            split_edge=split_edge,
            deep_data=data_splits["val"],
        )
        test_out = evaluate_data(
            type_="test",
            shallow=shallow,
            deep=deep,
            predictor=linkpredictor,
            batch_size=batch_size,
            split_edge=split_edge,
            deep_data=data_splits["test"],
        )
        shallow_predictions = {
            "train": {"y_pred_pos": train_out["pos_shallow"], "y_pred_neg": val_out["neg_shallow"]},
            "val": {"y_pred_pos": val_out["pos_shallow"], "y_pred_neg": val_out["neg_shallow"]},
            "test": {"y_pred_pos": test_out["pos_shallow"], "y_pred_neg": test_out["neg_shallow"]},
        }
        results_shallow = evaluator.collect_metrics(shallow_predictions)

        deep_predictions = {
            "train": {"y_pred_pos": train_out["pos_deep"], "y_pred_neg": val_out["neg_deep"]},
            "val": {"y_pred_pos": val_out["pos_deep"], "y_pred_neg": val_out["neg_deep"]},
            "test": {"y_pred_pos": test_out["pos_deep"], "y_pred_neg": test_out["neg_deep"]},
        }
        results_deep = evaluator.collect_metrics(deep_predictions)
        return (results_shallow, results_deep)


def warm_train(
    shallow,
    deep,
    predictor,
    data_deep,
    optimizer_shallow,
    optimizer_deep,
    data_shallow,
    criterion,
    batch_size=65538,
    config=None,
):
    shallow.train()
    deep.train()
    predictor.train()

    pos_edge_index = data_shallow
    pos_edge_index = pos_edge_index.to(data_deep.x.device)

    for perm in DataLoader(range(pos_edge_index.size(1)), batch_size, shuffle=True):
        optimizer_shallow.zero_grad()
        optimizer_deep.zero_grad()

        # W = deep(data_deep.x, data_deep.adj_t)
        W = (
            deep(data_deep.x, data_deep.adj_t)
            if config.dataset.dataset_name in ["ogbl-collab", "Flickr", "Twitch"]
            else deep(data_deep.x, data_deep.edge_index)
        )
        edge = pos_edge_index[:, perm]

        # shallow
        pos_out_shallow = shallow(edge[0], edge[1])

        # deep
        pos_out_deep = predictor(W[edge[0]], W[edge[1]])

        # Negative edges

        neg_edge_index = get_negative_samples(edge, data_deep.x.shape[0], edge.size(1))
        neg_edge_index = neg_edge_index.to(data_deep.x.device)
        neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])
        neg_out_deep = predictor(W[neg_edge_index[0]], W[neg_edge_index[1]])

        # concat positive and negative predictions
        total_predictions_shallow = torch.cat([pos_out_shallow, neg_out_shallow], dim=0)
        total_predictions_deep = torch.cat([pos_out_deep, neg_out_deep], dim=0)
        y_shallow = torch.cat(
            [torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0
        ).to(data_deep.x.device)
        y_deep = (
            torch.cat([torch.ones(pos_out_deep.size(0)), torch.zeros(neg_out_deep.size(0))], dim=0)
            .to(data_deep.x.device)
            .unsqueeze(1)
        )

        # calculate loss
        loss_shallow = criterion(total_predictions_shallow, y_shallow)
        loss_deep = criterion(total_predictions_deep, y_deep)

        # optimization step
        loss_shallow.backward()
        optimizer_shallow.step()

        # optimization step
        loss_deep.backward()

        torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer_deep.step()

    return (loss_shallow, loss_deep)


def fit_warm_start(
    warm_start,
    shallow,
    deep,
    data_deep,
    data_shallow,
    predictor,
    optimizer_shallow,
    optimizer_deep,
    criterion,
    split_edge,
    evaluator,
    batch_size,
    config,
    log,
    training_args,
    data_splits=None,
):
    prog_bar = tqdm(range(warm_start))
    for i in prog_bar:
        loss_shallow, loss_deep = warm_train(
            shallow=shallow,
            deep=deep,
            predictor=predictor,
            data_deep=data_deep,
            data_shallow=data_shallow,
            optimizer_shallow=optimizer_shallow,
            optimizer_deep=optimizer_deep,
            criterion=criterion,
            config=config,
            batch_size=training_args.batch_size,
        )

        prog_bar.set_postfix({"Shallow L": loss_shallow.item(), "Deep L": loss_deep.item()})

        if i % 10 == 0:
            results_shallow, results_deep = test_indi_with_predictor(
                shallow=shallow,
                deep=deep,
                linkpredictor=predictor,
                split_edge=split_edge,
                x=data_deep.x,
                adj_t=data_deep.adj_t
                if config.dataset.dataset_name in ["ogbl-collab", "Flickr", "Twitch"]
                else None,
                evaluator=evaluator,
                batch_size=batch_size,
                data_splits=data_splits,
                config=config,
            )

            log.info(
                f"Shallow: Train {config.dataset.track_metric}:{results_shallow['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_shallow['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_shallow['test'][config.dataset.track_metric]}"
            )
            log.info(
                f"Deep: Train {config.dataset.track_metric}:{results_deep['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_deep['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_deep['test'][config.dataset.track_metric]}"
            )


def fit_logits_integration(config, dataset, training_args, Logger, log, seeds, save_path):
    data = dataset[0]
    undirected = True
    if "edge_weight" in data:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    if data.is_directed():
        undirected = False
        data.edge_index = to_undirected(data.edge_index)

    data = data.to(config.device)
    data_shallow, split_edge, data_splits = get_split_edge(
        data=data, dataset=dataset, config=config, training_args=training_args
    )

    data_deep = T.ToSparseTensor()(data)
    if not undirected:  #
        data_deep.adj_t = data_deep.adj_t.to_symmetric()
    data_deep = data_deep.to(config.device)

    for counter, seed in enumerate(seeds):
        set_seed(seed)
        Logger.start_run()

        ##### Setup models #####
        init_embeddings = initialize_embeddings(
            num_nodes=data.x.shape[0],
            data=data_shallow,
            dataset=dataset,
            method=training_args.init,
            dim=training_args.embedding_dim,
            for_link=True,
            edge_split=split_edge,
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
        if training_args.deep_model == "GraphSage":
            deep = SAGE(
                in_channels=data.num_features,
                hidden_channels=training_args.deep_hidden_channels,
                out_channels=training_args.deep_out_dim,
                num_layers=training_args.deep_num_layers,
                dropout=training_args.deep_dropout,
            ).to(config.device)
        elif training_args.deep_model == "GCN":
            deep = GCN(
                in_channels=data.num_features,
                hidden_channels=training_args.deep_hidden_channels,
                out_channels=training_args.deep_out_dim,
                num_layers=training_args.deep_num_layers,
                dropout=training_args.deep_dropout,
            ).to(config.device)
        elif training_args.deep_model == "GAT":
            deep = GAT(
                in_channels=data.num_features,
                hidden_channels=training_args.deep_hidden_channels,
                out_channels=training_args.deep_out_dim,
                num_layers=training_args.deep_num_layers,
                dropout=training_args.deep_dropout,
            ).to(config.device)

        predictor = LinkPredictor(
            in_channels=training_args.deep_out_dim,
            hidden_channels=training_args.MLP_HIDDEN_CHANNELS,
            out_channels=1,
            num_layers=training_args.deep_num_layers,
            dropout=training_args.deep_dropout,
        ).to(config.device)

        # setup optimizer
        params_shallow = [
            {
                "params": shallow.parameters(),
                "lr": training_args.shallow_lr,
                "weight_decay": training_args.weight_decay_shallow,
            }
        ]
        params_deep = [
            {
                "params": list(deep.parameters()) + list(predictor.parameters()),
                "lr": training_args.deep_lr,
                "weight_decay": training_args.weight_decay_deep,
            }
        ]

        optimizer_shallow = torch.optim.Adam(params_shallow)
        optimizer_deep = torch.optim.Adam(params_deep)
        criterion = nn.BCEWithLogitsLoss()
        evaluator = METRICS(
            metrics_list=config.dataset.metrics, task="LinkPrediction", dataset=config.dataset.dataset_name
        )

        # Train warm start if provided
        if training_args.warm_start > 0:
            fit_warm_start(
                warm_start=training_args.warm_start,
                shallow=shallow,
                deep=deep,
                predictor=predictor,
                data_deep=data_deep
                if config.dataset.dataset_name in ["ogbl-collab", "Flickr", "Twitch"]
                else data_splits["train"],
                data_shallow=data_shallow,
                optimizer_shallow=optimizer_shallow,
                optimizer_deep=optimizer_deep,
                criterion=criterion,
                split_edge=split_edge,
                evaluator=evaluator,
                batch_size=training_args.batch_size,
                log=log,
                config=config,
                training_args=training_args,
                data_splits=data_splits,
            )
            shallow.train()
            deep.train()
            predictor.train()

        # Now consider joint traning
        λ = nn.Parameter(torch.tensor(training_args.lambda_))
        params_combined = [
            {
                "params": shallow.parameters(),
                "lr": training_args.shallow_lr_joint,
                "weight_decay": training_args.weight_decay_shallow,
            },
            {
                "params": list(deep.parameters()) + list(predictor.parameters()),
                "lr": training_args.deep_lr_joint,
                "weight_decay": training_args.weight_decay_deep,
            },
            {"params": [λ], "lr": training_args.lambda_lr},
        ]

        optimizer = torch.optim.Adam(params_combined)

        prog_bar = tqdm(range(training_args.joint_train))

        control_model_weights = ModelWeights(
            direction=training_args.direction,
            shallow_frozen_epochs=training_args.shallow_frozen_epochs,
            deep_frozen_epochs=training_args.deep_frozen_epochs,
        )

        for epoch in prog_bar:
            shallow.train()
            deep.train()
            predictor.train()

            control_model_weights.step(epoch=epoch, shallow=shallow, deep=deep, predictor=predictor)

            pos_edge_index = data_shallow
            pos_edge_index = pos_edge_index.to(config.device)

            loss_list = []
            for perm in DataLoader(range(pos_edge_index.size(1)), training_args.batch_size, shuffle=True):
                optimizer.zero_grad()
                W = (
                    deep(data_deep.x, data_deep.adj_t)
                    if config.dataset.dataset_name in ["ogbl-collab", "Flickr", "Twitch"]
                    else deep(data_splits["train"].x, data_splits["train"].edge_index)
                )

                edge = pos_edge_index[:, perm]

                # shallow
                pos_out_shallow = shallow(edge[0], edge[1])

                # deep
                pos_out_deep = predictor(W[edge[0]], W[edge[1]])

                # Negative edges
                neg_edge_index = get_negative_samples(edge, data_deep.x.shape[0], edge.size(1))
                neg_edge_index = neg_edge_index.to(data_deep.x.device)
                neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])
                neg_out_deep = predictor(W[neg_edge_index[0]], W[neg_edge_index[1]])

                # Now Predictions
                pos_logits = predict(
                    shallow_logit=pos_out_shallow,
                    lambda_=λ,
                    deep_logit=pos_out_deep.squeeze(),
                    training_args=training_args,
                )
                # total negative
                neg_logits = predict(
                    shallow_logit=neg_out_shallow,
                    lambda_=λ,
                    deep_logit=neg_out_deep.squeeze(),
                    training_args=training_args,
                )

                # concat positive and negative predictions
                total_predictions = torch.cat([pos_logits, neg_logits], dim=0)
                y = torch.cat(
                    [torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0
                ).to(config.device)

                # calculate loss
                loss = criterion(total_predictions, y)
                loss_list.append(loss.item())
                # optimization step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(shallow.parameters(), 1.0)
                optimizer.step()

            results = test_joint_with_predictor(
                shallow=shallow,
                deep=deep,
                linkpredictor=predictor,
                split_edge=split_edge,
                x=data_deep.x,
                adj_t=data_deep.adj_t,
                evaluator=evaluator,
                batch_size=training_args.batch_size,
                λ=λ,
                training_args=training_args,
                data_splits=data_splits,
                config=config,
            )
            prog_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "λ": λ.item(),
                    f"Train {config.dataset.track_metric}": results["train"][config.dataset.track_metric],
                    f"Val {config.dataset.track_metric}": results["val"][config.dataset.track_metric],
                    f"Test {config.dataset.track_metric}": results["test"][config.dataset.track_metric],
                }
            )
            Logger.add_to_run(loss=np.mean(loss_list), results=results)

        Logger.end_run()
    Logger.save_results(save_path + "/link_logits_integration.json")
    if "save_to_folder" in config:
        create_path(config.save_to_folder)
        additional_save_path = f"{config.save_to_folder}/{config.dataset.task}/{config.dataset.dataset_name}/{config.dataset.DIM}/{config.model_type}"
        create_path(f"{additional_save_path}")
        Logger.save_results(additional_save_path + f"/link_logits_integration_{training_args.deep_model}.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
