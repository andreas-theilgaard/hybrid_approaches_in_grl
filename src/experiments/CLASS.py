import argparse
import subprocess
from omegaconf import OmegaConf
import os

#BASE = "~/miniconda3/envs/act_DLG/bin/python src/experiments/run_exps.py --config-name='base.yaml'"
BASE = "python src/experiments/run_exps.py --config-name='base.yaml'"


DIMENSIONS = {2: 8, 8: 8, 16: 16, 64: 64, 128: 128}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script is used reproducing the results for a certain dataset"
    )
    # device
    parser.add_argument("--device", default="cuda", type=str, required=False, choices=["cpu", "cuda", "mps"])
    # runs
    parser.add_argument(
        "--runs",
        default=3,
        type=int,
        required=False,
        help="Number of times the experiments should be repeated",
    )
    args = parser.parse_args()
    print(args)

    for dataset in ["Flickr-class"]:
        for DIM in [16,128]:
            CONFIG_SETUP = f"{BASE} dataset={dataset} device={args.device} +save_to_folder='results'"
            conf = OmegaConf.load(f"src/config/dataset/{dataset}.yaml")
            print(conf)

            # ###########################
            # # BASELINE
            # # Run Baseline MLP Using Features
            subprocess.call(
                f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DIM={DIM} dataset.DownStream.saved_embeddings=False dataset.DownStream.using_features=True dataset.DownStream.training.hidden_channels={DIM}",
                shell=True,
            )

            # # # Run random
            # # # subprocess.call(
            # # #     f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DIM={DIM} dataset.DownStream.saved_embeddings=False dataset.DownStream.using_features=False dataset.DownStream.use_spectral=False dataset.DownStream.random=True dataset.DownStream.training.hidden_channels={DIMENSIONS[DIM]}",
            # # #     shell=True,
            # # # )

            # # # Run spectral method
            # # subprocess.call(
            # #     f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DIM={DIM} dataset.DownStream.saved_embeddings=False dataset.DownStream.using_features=False dataset.DownStream.use_spectral=True dataset.DownStream.training.hidden_channels={DIMENSIONS[DIM]}",
            # #     shell=True,
            # # )
            # ###########################

            # ###########################
            # # Run Node2Vec
            # subprocess.call(f"{CONFIG_SETUP} model_type='Node2Vec' runs=1 dataset.DIM={DIM}", shell=True)
            # # Define paths
            # Node2Vec_path = f"results/{conf.task}/{conf.dataset_name}/{DIM}/Node2Vec/Node2Vec_embedding.pth"
            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DIM={DIM} dataset.DownStream.saved_embeddings={Node2Vec_path} dataset.DownStream.using_features=False dataset.DownStream.training.hidden_channels={DIMENSIONS[DIM]}",
            #     shell=True,
            # )
            # ############################

            # # ############################
            # # # Run Shallow
            subprocess.call(
                f"{CONFIG_SETUP} model_type='Shallow' runs={args.runs} dataset.DIM={DIM}", shell=True
            )
            Shallow_path = f"results/{conf.task}/{conf.dataset_name}/{DIM}/Shallow"

            shallow_embeddings = (
                [x for x in os.listdir(Shallow_path) if (".pth" in x and "best" not in x)]
                if conf.task == "LinkPrediction"
                else [x for x in os.listdir(Shallow_path) if (".pth" in x)]
            )
            for shallow_embedding in shallow_embeddings:
                subprocess.call(
                    f"{CONFIG_SETUP} model_type='DownStream' runs=1 dataset.DIM={DIM} dataset.DownStream.saved_embeddings={Shallow_path+'/'+shallow_embedding} dataset.DownStream.using_features=False dataset.DownStream.training.hidden_channels={DIMENSIONS[DIM]}",
                    shell=True,
                )
            # # ############################

            # # ############################
            # # # Run GNN
            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.model='GraphSage'",
                shell=True,
            )
            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.model='GCN'",
            #     shell=True,
            # )

            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.model='GAT'",
            #     shell=True,
            # )

            # # Run GNN with Shallow
            # for shallow_embedding in shallow_embeddings:
            #     subprocess.call(
            #         f"{CONFIG_SETUP} model_type='GNN' runs=1 dataset.DIM={DIM} dataset.GNN.extra_info={Shallow_path+'/'+shallow_embedding} dataset.GNN.model='GraphSage'",
            #         shell=True,
            #     )
            #     subprocess.call(
            #         f"{CONFIG_SETUP} model_type='GNN' runs=1 dataset.DIM={DIM} dataset.GNN.extra_info={Shallow_path+'/'+shallow_embedding} dataset.GNN.model='GCN'",
            #         shell=True,
            #     )

            #     subprocess.call(
            #         f"{CONFIG_SETUP} model_type='GNN' runs=1 dataset.DIM={DIM} dataset.GNN.extra_info={Shallow_path+'/'+shallow_embedding} dataset.GNN.model='GAT'",
            #         shell=True,
            #     )

            # # Run GNN with spectral
            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.use_spectral=True dataset.GNN.model='GCN'",
            #     shell=True,
            # )
            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.use_spectral=True dataset.GNN.model='GraphSage'",
            #     shell=True,
            # )

            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.use_spectral=True dataset.GNN.model='GAT'",
            #     shell=True,
            # )

            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.extra_info={Node2Vec_path} dataset.GNN.model='GCN'",
            #     shell=True,
            # )
            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.extra_info={Node2Vec_path} dataset.GNN.model='GraphSage'",
            #     shell=True,
            # )

            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.extra_info={Node2Vec_path} dataset.GNN.model='GAT'",
            #     shell=True,
            # )

            # # ############################
            # # # GNN Direct
            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN_DIRECT' runs={args.runs} dataset.DIM={DIM} dataset.GNN_DIRECT.model='GraphSage'",
            #     shell=True,
            # )
            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN_DIRECT' runs={args.runs} dataset.DIM={DIM} dataset.GNN_DIRECT.model='GCN'",
            #     shell=True,
            # )

            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='GNN_DIRECT' runs={args.runs} dataset.DIM={DIM} dataset.GNN_DIRECT.model='GAT'",
            #     shell=True,
            # )

            # # # Prediction for GNN Direct
            # GNN_DIRECT_PATH = f"results/{conf.task}/{conf.dataset_name}/{DIM}/GNN_DIRECT/models/"
            # for file in os.listdir(GNN_DIRECT_PATH):
            #     subprocess.call(
            #         f"{CONFIG_SETUP} model_type='DownStream' runs=1 dataset.DIM={DIM} dataset.DownStream.saved_embeddings={GNN_DIRECT_PATH+'/'+file} dataset.DownStream.using_features=False dataset.DownStream.training.hidden_channels={DIMENSIONS[DIM]}",
            #         shell=True,
            #     )

            # ############################
            # # Run combined method
            subprocess.call(
                f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='comb1' dataset.combined.comb1.training.MLP_HIDDEN={DIMENSIONS[DIM]} dataset.combined.comb1.training.deep_model='GraphSage'",
                shell=True,
            )


            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='comb1' dataset.combined.comb1.training.MLP_HIDDEN={DIMENSIONS[DIM]} dataset.combined.comb1.training.deep_model='GCN'",
            #     shell=True,
            # )

            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='comb1' dataset.combined.comb1.training.MLP_HIDDEN={DIMENSIONS[DIM]} dataset.combined.comb1.training.deep_model='GAT'",
            #     shell=True,
            # )



            subprocess.call(
                f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='comb2' dataset.combined.comb2.training.MLP_HIDDEN={DIMENSIONS[DIM]} dataset.combined.comb2.training.deep_model='GraphSage'",
                shell=True,
            )
            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='comb2' dataset.combined.comb2.training.MLP_HIDDEN={DIMENSIONS[DIM]} dataset.combined.comb2.training.deep_model='GCN'",
            #     shell=True,
            # )
            # subprocess.call(
            #     f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='comb2' dataset.combined.comb2.training.MLP_HIDDEN={DIMENSIONS[DIM]} dataset.combined.comb2.training.deep_model='GAT'",
            #     shell=True,
            # )
            # ############################
