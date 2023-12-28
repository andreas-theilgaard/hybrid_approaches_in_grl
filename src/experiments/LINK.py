import argparse
import subprocess
from omegaconf import OmegaConf
import os

BASE = "python src/experiments/run_model.py --config-name='base.yaml'"

IDENTIFIER = "QUICK_TEST"


DIMENSIONS = {2: 8, 8: 8, 16: 16, 64: 64, 128: 128}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script is used reproducing the results for a certain dataset"
    )
    # device
    parser.add_argument("--device", default="cpu", type=str, required=False, choices=["cpu", "cuda", "mps"])
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

    for dataset in ["Flickr-link"]:
        for DIM in [16]:
            CONFIG_SETUP = f"{BASE} dataset={dataset} device={args.device} identifier={IDENTIFIER} +save_to_folder='results'"
            conf = OmegaConf.load(f"src/config/dataset/{dataset}.yaml")
            print(conf)

            ###########################
            #BASELINE
            #Run Baseline MLP Using Features
            subprocess.call(
                f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DIM={DIM} dataset.DownStream.saved_embeddings=False dataset.DownStream.using_features=True dataset.DownStream.training.hidden_channels={DIM}",
                shell=True,
            )

            # # #Run spectral method
            subprocess.call(
                f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DIM={DIM} dataset.DownStream.saved_embeddings=False dataset.DownStream.using_features=False dataset.DownStream.use_spectral=True dataset.DownStream.training.hidden_channels={DIMENSIONS[DIM]}",
                shell=True,
            )
            # ###########################


            ############################
            #Run Node2Vec
            subprocess.call(f"{CONFIG_SETUP} model_type='Node2Vec' runs=1 dataset.DIM={DIM}", shell=True)
            #Define paths
            Node2Vec_path = f"results/{conf.task}/{conf.dataset_name}/{DIM}/Node2Vec/Node2Vec_embedding.pth"
            subprocess.call(
                f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DIM={DIM} dataset.DownStream.saved_embeddings={Node2Vec_path} dataset.DownStream.using_features=False dataset.DownStream.training.hidden_channels={DIMENSIONS[DIM]}",
                shell=True,
            )
            #############################

            ###############################
            #Run Shallow
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
            # # #############################

            # # # ############################
            #Run GNN
            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.model='GraphSage'",
                shell=True,
            )
            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.model='GCN'",
                shell=True,
            )

            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.model='GAT'",
                shell=True,
            )

            # # #Run GNN with Shallow
            for shallow_embedding in shallow_embeddings:
                subprocess.call(
                    f"{CONFIG_SETUP} model_type='GNN' runs=1 dataset.DIM={DIM} dataset.GNN.extra_info={Shallow_path+'/'+shallow_embedding} dataset.GNN.model='GraphSage'",
                    shell=True,
                )
                subprocess.call(
                    f"{CONFIG_SETUP} model_type='GNN' runs=1 dataset.DIM={DIM} dataset.GNN.extra_info={Shallow_path+'/'+shallow_embedding} dataset.GNN.model='GCN'",
                    shell=True,
                )
                subprocess.call(
                    f"{CONFIG_SETUP} model_type='GNN' runs=1 dataset.DIM={DIM} dataset.GNN.extra_info={Shallow_path+'/'+shallow_embedding} dataset.GNN.model='GAT'",
                    shell=True,
                )

            # # # # Run GNN with spectral
            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.use_spectral=True dataset.GNN.model='GCN'",
                shell=True,
            )
            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.use_spectral=True dataset.GNN.model='GraphSage'",
                shell=True,
            )

            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.use_spectral=True dataset.GNN.model='GAT'",
                shell=True,
            )


            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.extra_info={Node2Vec_path} dataset.GNN.model='GCN'",
                shell=True,
            )
            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.extra_info={Node2Vec_path} dataset.GNN.model='GraphSage'",
                shell=True,
            )

            subprocess.call(
                f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.DIM={DIM} dataset.GNN.extra_info={Node2Vec_path} dataset.GNN.model='GAT'",
                shell=True,
            )

            
            #############################
            #Run combined method

            # # Logits Integration
            subprocess.call(
                f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='LogitsIntegration' dataset.combined.LogitsIntegration.training.deep_model='GraphSage' dataset.combined.LogitsIntegration.training.MLP_HIDDEN_CHANNELS={DIMENSIONS[DIM]}",
                shell=True,
            )

            subprocess.call(
                f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='LogitsIntegration' dataset.combined.LogitsIntegration.training.deep_model='GCN' dataset.combined.LogitsIntegration.training.MLP_HIDDEN_CHANNELS={DIMENSIONS[DIM]}",
                shell=True,
            )

            subprocess.call(
                f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='LogitsIntegration' dataset.combined.LogitsIntegration.training.deep_model='GAT' dataset.combined.LogitsIntegration.training.MLP_HIDDEN_CHANNELS={DIMENSIONS[DIM]}",
                shell=True,
            )

            # # combined Latents
            subprocess.call(
                f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='CombinedLatents' dataset.combined.CombinedLatents.training.MLP_HIDDEN_CHANNELS={DIMENSIONS[DIM]} dataset.combined.CombinedLatents.training.deep_model='GraphSage'",
                shell=True,
            )

            subprocess.call(
                f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='CombinedLatents' dataset.combined.CombinedLatents.training.MLP_HIDDEN_CHANNELS={DIMENSIONS[DIM]} dataset.combined.CombinedLatents.training.deep_model='GCN'",
                shell=True,
            )

            subprocess.call(
                f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.DIM={DIM} dataset.combined.type='CombinedLatents' dataset.combined.CombinedLatents.training.MLP_HIDDEN_CHANNELS={DIMENSIONS[DIM]} dataset.combined.CombinedLatents.training.deep_model='GAT'",
                shell=True,
            )
