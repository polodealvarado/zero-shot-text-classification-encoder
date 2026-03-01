"""
Model registry for zero-shot text classification variants.

All models follow the same interface:
    - forward(texts, batch_labels) -> (scores [B, max_labels], mask [B, max_labels])
    - compute_loss(scores, targets, mask) -> scalar
    - predict(texts, labels) -> list[dict]
    - Inherits PyTorchModelHubMixin for save/load/push_to_hub
"""

from models.base import BiEncoderModel
from models.convmatch import ConvMatchModel
from models.dynquery import DynQueryModel
from models.late_interaction import LateInteractionModel
from models.polyencoder import PolyEncoderModel
from models.projection import ProjectionBiEncoderModel
from models.spanclass import SpanClassModel

MODEL_REGISTRY = {
    "biencoder": BiEncoderModel,
    "projection_biencoder": ProjectionBiEncoderModel,
    "late_interaction": LateInteractionModel,
    "polyencoder": PolyEncoderModel,
    "dynquery": DynQueryModel,
    "spanclass": SpanClassModel,
    "convmatch": ConvMatchModel,
}

__all__ = [
    "MODEL_REGISTRY",
    "BiEncoderModel",
    "ProjectionBiEncoderModel",
    "LateInteractionModel",
    "PolyEncoderModel",
    "DynQueryModel",
    "SpanClassModel",
    "ConvMatchModel",
]
