"""fastembed-backed EmbeddingModel. Requires `fastembed` (not needed for tests)."""
from __future__ import annotations

from fastembed import TextEmbedding  # type: ignore

from ..contracts.models import Vector


class FastEmbedModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    @property
    def dim(self) -> int:
        return self.model._get_model_description(self.model_name).dim

    def encode(self, text: str) -> Vector:
        return next(iter(self.model.embed(text))).tolist()
