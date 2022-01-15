import os

import torch

from embeddings.embedding_model import EmbeddingModel


class Embedder:
    def __init__(self, device, kwargs):
        self.embedding_model = EmbeddingModel(kwargs).to(device)
        self.device = device
        self._load_model()

    def _load_model(
        self,
        models_folder: str = "embeddings/pretrained",
        model_name: str = "EmbeddingModel",
    ):
        checkpoint = torch.load(os.path.join(models_folder, model_name))
        self.embedding_model.load_state_dict(checkpoint["model_state_dict"])

    def get_embedding(
        self,
        features: torch.Tensor,
        embedds: torch.Tensor,
        sections: torch.Tensor,
    ):
        return self.embedding_model(
            features.to(self.device), embedds.to(self.device), sections.to(self.device)
        )
