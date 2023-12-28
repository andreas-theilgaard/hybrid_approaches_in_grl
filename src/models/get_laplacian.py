from src.data.get_data import DataLoader
from torch_geometric.utils import to_undirected
from src.models.model_utils import get_k_laplacian_eigenvectors
import torch
from src.models.model_utils import create_path


def get_laplacian(K, task, dataset_name, edge_split=None, config=None):
    dataset = DataLoader(model_name="Shallow", task_type=task, dataset=dataset_name).get_data()
    data = dataset[0]
    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)
    if task == "NodeClassification":
        eigen_vectors = get_k_laplacian_eigenvectors(
            num_nodes=data.x.shape[0], data=data, dataset=dataset, k=K, for_link=False
        )
    else:
        eigen_vectors = get_k_laplacian_eigenvectors(
            num_nodes=data.x.shape[0], data=data, dataset=dataset, k=K, for_link=True, edge_split=edge_split
        )
    if "save_to_folder" in config:
        create_path(config.save_to_folder)
        additional_save_path = f"{config.save_to_folder}/{config.dataset.task}/{config.dataset.dataset_name}/{config.dataset.DIM}/Laplacian"
        create_path(f"{additional_save_path}")
        torch.save(eigen_vectors, additional_save_path + "/laplacian.pth")
    return eigen_vectors
