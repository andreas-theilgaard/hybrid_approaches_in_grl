from src.data.get_data import DataLoader
import numpy as np
from src.models.model_utils import get_k_laplacian_eigenvectors
import torch_geometric.transforms as T
from tqdm import tqdm
from src.data.data_utils import get_link_data_split

dataset = DataLoader(model_name='DownStream',task_type='LinkPrediction',dataset='Flickr').get_data()
data = dataset[0]


train_data, val_data, test_data = get_link_data_split(data, dataset_name='Flickr')
split_edge = {
    "train": {"edge": train_data.pos_edge_label_index.T},
    "valid": {
        "edge": val_data.pos_edge_label_index.T,
        "edge_neg": val_data.neg_edge_label_index.T,
    },
    "test": {
        "edge": test_data.pos_edge_label_index.T,
        "edge_neg": test_data.neg_edge_label_index.T,
    },
}


def is_in_edgeindex(val1,val2):
    left_idx = torch.where(train_data.edge_index.t()[:,0]==val1)[0]
    right_idx = torch.where(train_data.edge_index.t()[:,1]==val2)[0]
    res = list(set(left_idx.tolist()) & set(right_idx.tolist()))
    if len(res)>0:
        return 1
    return 0


total_true = 0
for i in tqdm(range(1000)):
    edges = split_edge['train']['edge'][i]
    total_true+= is_in_edgeindex(edges[0].item(),edges[1].item())



train_data.edge_index.shape
val_data.edge_index.shape
test_data.edge_index.shape

data.edge_index.shape

# data = T.ToSparseTensor()(data)
#split_edge = dataset.get_edge_split()






import torch

def is_in_edgeindex(val1,val2):
    left_idx = torch.where(data.edge_index.t()[:,0]==val1)[0]
    right_idx = torch.where(data.edge_index.t()[:,1]==val2)[0]
    res = list(set(left_idx.tolist()) & set(right_idx.tolist()))
    if len(res)>0:
        return 1
    return 0
    

total_true = 0
for i in tqdm(range(10000)):
    edges = split_edge['test']['edge'][i]
    total_true+= is_in_edgeindex(edges[0].item(),edges[1].item())




split_edge['train']['edge'].shape[0]



right_idx.tolist()


data.adj_t[64257,45227]

data.adj_t

split_edge['test']['edge'].shape

for i in range(100):
    edges = split_edge['test']['edge'][i]
    print(edges[0].item(),edges[1].item(),data.adj_t[edges[0].item(),edges[1].item()])



for i in range(5):
    #set_seed(seed)
    for dim in [2,8,16,128]:
        x = get_k_laplacian_eigenvectors(
                data=data,
                dataset=dataset,
                k=dim,
                is_undirected=True,
                for_link=False,
                edge_split=None,
                num_nodes=data.x.shape[0],
            )
        print(x.shape)


# import torch
# import torch_geometric
# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
# from scipy.sparse.csgraph import laplacian
# from scipy.linalg import eigh
# import numpy as np

# dataset = Planetoid(root='data', name='Cora', transform=T.NormalizeFeatures())
# data = dataset[0]
# def construct_laplacian(data):
#     # Get the adjacency matrix in COO format
#     edge_index = data.edge_index.numpy()
#     num_nodes = data.num_nodes
#     adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

#     # Fill in the adjacency matrix
#     adjacency_matrix[edge_index[0], edge_index[1]] = 1
#     adjacency_matrix[edge_index[1], edge_index[0]] = 1  # Ensure symmetry

#     # Convert to scipy sparse matrix
#     adjacency_matrix = adjacency_matrix.numpy()
#     laplacian_matrix = laplacian(adjacency_matrix, normed=True)
#     return laplacian_matrix






