import torch
from torch_geometric.nn import Node2Vec as N2V
from tqdm import tqdm
from src.models.model_utils import create_path


class Node2Vec:
    def __init__(
        self,
        edge_index,
        device,
        config,
        save_path: str,
        embedding_dim: int = 128,
        walk_length: int = 80,
        walks_per_node: int = 10,
        context_size: int = 20,
        num_negative_samples: int = 1,
        sparse=True,
    ):
        self.save_path = save_path
        self.device = device
        self.model = N2V(
            edge_index=edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            sparse=sparse,
        ).to(device)
        self.config = config

    def save_embeddings(self, model):
        self.embedding_save_path = self.save_path + "/embedding.pth"
        torch.save(model.embedding.weight.data.cpu(), self.embedding_save_path)

    def train(self, batch_size, epochs, lr, num_workers=0):
        loader = self.model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
        optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=lr)
        self.model.train()
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            for i, (pos_sample, neg_sample) in enumerate(loader):
                optimizer.zero_grad()
                loss = self.model.loss(pos_sample.to(self.device), neg_sample.to(self.device))
                loss.backward()
                optimizer.step()
                prog_bar.set_postfix({"loss": loss.item()})
            # Save embedding
        self.save_embeddings(model=self.model)
        print(
            f"Embeddings have been saved at {self.embedding_save_path} you can now use them for any downstream task"
        )
        if "save_to_folder" in self.config:
            create_path(self.config.save_to_folder)
            additional_save_path = f"{self.config.save_to_folder}/{self.config.dataset.task}/{self.config.dataset.dataset_name}/{self.config.dataset.DIM}/{self.config.model_type}"
            create_path(additional_save_path)
            torch.save(
                self.model.embedding.weight.data.cpu(), additional_save_path + "/Node2Vec_embedding.pth"
            )

        return self.model
