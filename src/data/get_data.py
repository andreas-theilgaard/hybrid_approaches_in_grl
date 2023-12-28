from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from src.data.data_utils import TorchGeometricDatasets

class DataLoader:
    def __init__(
        self,
        model_name: str,
        task_type: str = "NodeClassification",
        dataset: str = "ogbn-arxiv",
        log=None,
    ):
        self.task_type = task_type
        self.dataset = dataset
        self.model_name = model_name
        self.log = log
        self.assert_arguments()

    def assert_arguments(self):
        assert self.task_type in [
            "NodeClassification",
            "LinkPrediction",
        ], f"Expect task_type to be either 'NodeClassification' or 'LinkPrediction' but received {self.task_type}"

        if self.dataset in ["ogbn-arxiv", "ogbn-products", "ogbn-mag"] and self.task_type == "LinkPrediction":
            raise ValueError(f"{self.dataset} can only be used for NodeClassification")
        if (
            self.dataset in ["ogbl-collab", "ogbl-ppa", "ogbl-vessel", "ogbl-citation2"]
            and self.task_type == "NodeClassification"
        ):
            raise ValueError(f"{self.dataset} can only be used for LinkPrediction")

    def get_NodeClassification_dataset(self):
        if self.dataset in ["ogbn-arxiv"]:
            dataset = (
                PygNodePropPredDataset(name=self.dataset, root="data")
                if self.model_name not in ["GNN"]
                else PygNodePropPredDataset(name=self.dataset, root="data", transform=T.ToSparseTensor())
            )
        elif self.dataset == "ogbn-mag":
            dataset = PygNodePropPredDataset(name=self.dataset, root="data")
        elif self.dataset in ["Cora", "CiteSeer", "PubMed", "Flickr", "Computers", "Photo", "Twitch"]:
            PYG_DATASETS = TorchGeometricDatasets(
                dataset=self.dataset, task=self.task_type, model=self.model_name
            )
            dataset = PYG_DATASETS.get_dataset()
        self.dataset_summary(dataset)
        return dataset

    def get_LinkPrediction_dataset(self):
        if self.dataset in ["ogbl-collab", "ogbl-ppa", "ogbl-vessel", "ogbl-citation2"]:
            dataset = PygLinkPropPredDataset(name=self.dataset, root="data")
        elif self.dataset in ["Cora", "CiteSeer", "PubMed", "Flickr", "Computers", "Photo", "Twitch"]:
            PYG_DATASETS = TorchGeometricDatasets(
                dataset=self.dataset, task=self.task_type, model=self.model_name
            )
            dataset = PYG_DATASETS.get_dataset()
        self.dataset_summary(dataset)
        return dataset

    def dataset_summary(self, dataset):
        summary = f"""\n    ===========================
    Dataset: {self.dataset}:
    ===========================
    Number of graphs: {len(dataset)} \n
    Number of features: {dataset.num_features} \n
        """
        if self.task_type == "NodeClassification":
            summary += f"Number of classes: {dataset.num_classes}"
        if len(dataset) == 1:
            data = dataset[0]
            summary += f"""
    Number of nodes: {data.num_nodes} \n
    Number of edges: {data.num_edges} \n"""
        if self.dataset != "ogbn-mag":
            summary += f"""Is undirected: {data.is_undirected()}"""
        if self.log:
            self.log.info(summary)

    def get_data(self):
        if self.task_type == "NodeClassification":
            return self.get_NodeClassification_dataset()
        elif self.task_type == "LinkPrediction":
            return self.get_LinkPrediction_dataset()
