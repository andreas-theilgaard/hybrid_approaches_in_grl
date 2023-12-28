import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from src.models.model_utils import set_seed, get_seeds, get_negative_samples
from src.models.LinkPrediction.combined.combined_link_utils import decode, predict, ModelWeights
from src.models.shallow import ShallowModel, initialize_embeddings
from src.models.metrics import METRICS
from src.models.model_utils import prepare_metric_cols
import copy
from src.models.NodeClassification.combined.combined_class_utils import NodeClassifier, test_MLP, SAGE, GCN, GAT
from torch_geometric.data import Data
from src.models.model_utils import create_path


def warm_train(
    shallow, deep, data_deep, optimizer_shallow, optimizer_deep, data, criterion, gamma, training_args
):
    shallow.train()
    deep.train()
    optimizer_shallow.zero_grad()
    optimizer_deep.zero_grad()

    pos_edge_index = data.edge_index
    pos_edge_index = pos_edge_index.to(data_deep.x.device)
    neg_edge_index = get_negative_samples(pos_edge_index, data_deep.x.shape[0], pos_edge_index.size(1))
    neg_edge_index = neg_edge_index.to(data_deep.x.device)

    #######################
    # Shallow
    #######################
    # Positive edges
    pos_out_shallow = shallow(pos_edge_index[0], pos_edge_index[1])
    # Negative edges
    neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])

    #######################
    # Deep
    #######################
    # Get deep embeddings
    # Positive edges

    W = deep(data_deep.x, data_deep.adj_t)
    pos_out_deep = decode(
        W=W, node_i=pos_edge_index[0], node_j=pos_edge_index[1], gamma=gamma, type_=training_args.deep_decode
    )
    # Negative edges
    neg_out_deep = decode(
        W=W, node_i=neg_edge_index[0], node_j=neg_edge_index[1], gamma=gamma, type_=training_args.deep_decode
    )

    # concat positive and negative predictions
    total_predictions_shallow = torch.cat([pos_out_shallow, neg_out_shallow], dim=0)
    total_predictions_deep = torch.cat([pos_out_deep, neg_out_deep], dim=0)
    y_shallow = torch.cat(
        [torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0
    ).to(data_deep.x.device)
    y_deep = torch.cat([torch.ones(pos_out_deep.size(0)), torch.zeros(neg_out_deep.size(0))], dim=0).to(
        data_deep.x.device
    )

    # calculate loss
    loss_shallow = criterion(total_predictions_shallow, y_shallow)
    loss_deep = criterion(total_predictions_deep, y_deep)

    # optimization step
    loss_shallow.backward()
    optimizer_shallow.step()

    # optimization step
    loss_deep.backward()
    optimizer_deep.step()
    return (loss_shallow, loss_deep)


def fit_warm_start(
    warm_start,
    shallow,
    deep,
    data_deep,
    data,
    optimizer_shallow,
    optimizer_deep,
    criterion,
    gamma,
    training_args,
):
    prog_bar = tqdm(range(warm_start))
    for i in prog_bar:
        loss_shallow, loss_deep = warm_train(
            shallow=shallow,
            deep=deep,
            data_deep=data_deep,
            data=data,
            optimizer_shallow=optimizer_shallow,
            optimizer_deep=optimizer_deep,
            criterion=criterion,
            gamma=gamma,
            training_args=training_args,
        )

        prog_bar.set_postfix({"Shallow L": loss_shallow.item(), "Deep L": loss_deep.item()})


def fit_logits_integration(config, dataset, training_args, Logger, log, seeds, save_path):
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

        ##### Setup models #####
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

        # setup optimizer
        params_shallow = [{"params": shallow.parameters(), "lr": training_args.shallow_lr}]
        gamma = nn.Parameter(torch.tensor(training_args.gamma))
        params_deep = [{"params": list(deep.parameters()) + [gamma], "lr": training_args.deep_lr}]

        optimizer_shallow = torch.optim.Adam(params_shallow)
        optimizer_deep = torch.optim.Adam(params_deep)
        criterion = nn.BCEWithLogitsLoss()
        evaluator = METRICS(
            metrics_list=config.dataset.metrics,
            task="NodeClassification",
            dataset=config.dataset.dataset_name,
        )

        # Train warm start if provided
        if training_args.warm_start > 0:
            fit_warm_start(
                warm_start=training_args.warm_start,
                shallow=shallow,
                deep=deep,
                data_deep=data_deep,
                data=data,
                optimizer_shallow=optimizer_shallow,
                optimizer_deep=optimizer_deep,
                criterion=criterion,
                gamma=gamma,
                training_args=training_args,
            )
            shallow.train()
            deep.train()

        # Now consider joint traning
        λ = nn.Parameter(torch.tensor(training_args.lambda_))

        DEEP_PARAMS = {"params": list(deep.parameters()) + [gamma], "lr": training_args.deep_lr_joint}

        params_combined = [
            {"params": shallow.parameters(), "lr": training_args.shallow_lr_joint},
            DEEP_PARAMS,
            {"params": [λ], "lr": training_args.lambda_lr},
        ]

        optimizer = torch.optim.Adam(params_combined)
        control_model_weights = ModelWeights(
            direction=training_args.direction,
            shallow_frozen_epochs=training_args.shallow_frozen_epochs,
            deep_frozen_epochs=training_args.deep_frozen_epochs,
        )

        prog_bar = tqdm(range(training_args.joint_train))
        for epoch in prog_bar:
            shallow.train()
            deep.train()
            control_model_weights.step(epoch=epoch, shallow=shallow, deep=deep)

            optimizer.zero_grad()

            pos_edge_index = data.edge_index
            pos_edge_index = pos_edge_index.to(config.device)
            neg_edge_index = get_negative_samples(
                pos_edge_index, data_deep.x.shape[0], pos_edge_index.size(1)
            )
            neg_edge_index = neg_edge_index.to(config.device)

            #######################
            # Shallow
            #######################
            # Positive edges
            pos_out_shallow = shallow(pos_edge_index[0], pos_edge_index[1])
            # Negative edges
            neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])

            #######################
            # Deep
            #######################
            # Get deep embeddings
            # Positive edges
            W = deep(data_deep.x, data_deep.adj_t)
            pos_out_deep = decode(
                W=W,
                node_i=pos_edge_index[0],
                node_j=pos_edge_index[1],
                gamma=gamma,
                type_=training_args.deep_decode,
            )
            # Negative edges
            neg_out_deep = decode(
                W=W,
                node_i=neg_edge_index[0],
                node_j=neg_edge_index[1],
                gamma=gamma,
                type_=training_args.deep_decode,
            )

            # # Combine model predictions

            # total positive
            pos_logits = predict(
                shallow_logit=pos_out_shallow, deep_logit=pos_out_deep, lambda_=λ, training_args=training_args
            )
            # total negative
            neg_logits = predict(
                shallow_logit=neg_out_shallow, deep_logit=neg_out_deep, lambda_=λ, training_args=training_args
            )

            # concat positive and negative predictions
            total_predictions = torch.cat([pos_logits, neg_logits], dim=0)
            y = torch.cat(
                [torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0
            ).to(config.device)

            # calculate loss
            loss = criterion(total_predictions, y)

            # optimization step
            loss.backward()
            optimizer.step()
            prog_bar.set_postfix({"Shallow L": loss.item(), "λ": λ.item()})

        # Now NodeClassificaation training
        shallow.train()
        deep.train()
        deep_emb = deep(data_deep.x, data_deep.adj_t).detach().cpu()
        shallow_emb = shallow.embeddings.weight.data.detach().cpu()
        deep_emb = deep_emb.to(config.device)
        shallow_emb = shallow_emb.to(config.device)
        concat_embeddings = torch.cat([shallow_emb, deep_emb], dim=-1).to(config.device)

        Classifier = NodeClassifier(
            in_channels=deep_emb.shape[1] + shallow_emb.shape[1],
            hidden_channels=training_args.MLP_HIDDEN,
            out_channels=dataset.num_classes,
            num_layers=training_args.MLP_NUM_LAYERS,
            dropout=training_args.MLP_DROPOUT,
        ).to(config.device)

        MLP_optimizer = torch.optim.Adam(Classifier.parameters(), lr=training_args.MLP_LR)

        criterion = torch.nn.CrossEntropyLoss()
        train_idx = split_idx["train"].to(config.device)
        y = data.y.to(config.device)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        for epoch in tqdm(range(training_args.MLP_EPOCHS)):
            Classifier.train()
            MLP_optimizer.zero_grad()
            out = Classifier(concat_embeddings)
            loss = criterion(
                out[train_idx], (y[train_idx]).type(torch.LongTensor).to(config.device).squeeze(1)
            )
            loss.backward()
            MLP_optimizer.step()
            results = test_MLP(
                model=Classifier,
                x=concat_embeddings,
                y=y,
                split_idx=split_idx,
                evaluator=evaluator,
                config=config,
            )
            Logger.add_to_run(loss=loss.item(), results=results)
            # print(results['train']['acc'],results['test']['acc'])

        Logger.end_run()
    Logger.save_results(save_path + "/class_logits_integration.json")

    if "save_to_folder" in config:
        create_path(config.save_to_folder)
        additional_save_path = f"{config.save_to_folder}/{config.dataset.task}/{config.dataset.dataset_name}/{config.dataset.DIM}/{config.model_type}"
        create_path(f"{additional_save_path}")
        Logger.save_results(additional_save_path + f"/class_logits_integration_{training_args.deep_model}.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
