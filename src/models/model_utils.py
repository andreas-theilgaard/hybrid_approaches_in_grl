import torch
from datetime import datetime
from pathlib import Path
import random
import numpy as np
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_geometric.utils import negative_sampling

def get_negative_samples(edge_index, num_nodes, num_neg_samples):
    return negative_sampling(edge_index, num_neg_samples=num_neg_samples, num_nodes=num_nodes)


def prepare_metric_cols(metrics):
    metric_cols = ["Train loss"]
    for metric in metrics:
        for col in ["Train", "Val", "Test"]:
            if metric == "loss" and col != "Train":
                continue
            else:
                metric_cols.append(f"{col} {metric}")
    return metric_cols


def get_seeds(n=10):
    return list(range(n))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_best_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def create_path(save_path):
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)


def cur_time():
    return (datetime.now()).strftime("%b-%d %H:%M")


def get_k_laplacian_eigenvectors(
    num_nodes,
    data,
    dataset,
    k,
    is_undirected: bool = True,
    SPARSE_THRESHOLD: int = 5000,
    for_link: bool = False,
    edge_split=None,
):
    """
    Computes the k first non-trivial eigenvectors for the normalized laplacian matrix
    args:
        data:
            the graph data
        dataset:
            the dataset for the graph
        k:
            the number of non-trivial eigenvectors to return
        is_undirected:
            whether graph data is_indirected or not
        SPARSE_THRESHOLD:
            number of nodes used for sparse eigenvector computation
        for_link:
            Flag used for link prediciton task
    Return:
        k first non-trivial eigenvectors
    """
    # print("Getting Laplacian eigenvectors")
    num_nodes = num_nodes
    is_undirected = True
    if for_link:
        try:
            if isinstance(edge_split["train"]["weight"], torch.Tensor):
                edge_weight_in = ((edge_split["train"]["weight"])).float()
                if len(edge_weight_in.shape) == 1:
                    edge_weight_in = edge_weight_in.unsqueeze(1)
            else:
                edge_weight_in = None
        except:
            edge_weight_in = None

        edge_index, edge_weight = get_laplacian(
            data, edge_weight_in, normalization="sym", num_nodes=num_nodes
        )
    else:
        edge_weight_in = data.edge_weight
        edge_weight_in = edge_weight_in.float() if edge_weight_in else edge_weight_in
        edge_index, edge_weight = get_laplacian(
            data.edge_index, edge_weight_in, normalization="sym", num_nodes=num_nodes
        )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    if num_nodes < SPARSE_THRESHOLD:
        from numpy.linalg import eig, eigh

        eig_fn = eig if not is_undirected else eigh
        eig_vals, eig_vecs = eig_fn(L.todense())
    else:
        from scipy.sparse.linalg import eigs, eigsh

        eig_fn = eigs if not is_undirected else eigsh
        eig_vals, eig_vecs = eig_fn(
            L,
            k=k + 1,
            which="SR" if not is_undirected else "SA",
            return_eigenvectors=True,
            maxiter=num_nodes*100
        )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    # exclude first trivial eigenvector
    eigen_vectors = torch.from_numpy(eig_vecs[:, 1 : k + 1])
    # print("Returning Laplacian eigenvectors")
    return eigen_vectors
