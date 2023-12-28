from torch_geometric.datasets import Planetoid, Flickr, Amazon, Twitch
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import to_undirected
from src.models.model_utils import set_seed
import torch_geometric.transforms as T
from torch_geometric.transforms import NormalizeFeatures


def get_link_data_split(data, dataset_name: str, num_test: float = 0.1, num_val: float = 0.05):
    set_seed(1234)
    if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        num_test = 0.1
        num_val = 0.05
    elif dataset_name in ["Flickr", "Twitch"]:
        num_test = 0.2
        num_val = 0.1
    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)
    transform = RandomLinkSplit(
        is_undirected=True,
        num_test=num_test,
        num_val=num_val,
        add_negative_train_samples=False,
        split_labels=True,
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data


class TorchGeometricDatasets:
    def __init__(self, dataset: str, task: str, model: str):
        self.dataset = dataset
        self.task = task
        self.model = model

    def get_dataset(self):
        if self.dataset in ["Cora", "CiteSeer", "PubMed"]:
            dataset = Planetoid(root="data", name=self.dataset, transform=self.use_transform())
        elif self.dataset in ["Flickr"]:
            dataset = Flickr(root="data", transform=self.use_transform())
        elif self.dataset in ["Computers", "Photo"]:
            dataset = Amazon(root="data", name=self.dataset, transform=self.use_transform())
        elif self.dataset in ["Twitch"]:
            dataset = Twitch(root="data", name="EN", transform=self.use_transform())
        return dataset

    def use_transform(self):
        set_seed(1234)
        if self.dataset in ["Cora"]:
            return NormalizeFeatures()
        if self.dataset in ["Twitch"]:
            return RandomNodeSplit(split="train_rest", num_test=1000, num_val=500)
        return None
