from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attr_config import AttributionConfig


class HeatmapUtils:
    @staticmethod
    def normalize(t: torch.Tensor) -> torch.Tensor:
        t = t.clone()
        dims = tuple(range(1, t.ndim))  # Preserve the batch dimension
        t_min = t.amin(dim=dims, keepdim=True)
        t_max = t.amax(dim=dims, keepdim=True)
        return (t - t_min) / (t_max - t_min + 1e-8)

    @staticmethod
    def upsample(
        heatmap: torch.Tensor, shape: Tuple[int, int], mode: str = "bilinear"
    ) -> torch.Tensor:
        return F.interpolate(heatmap, size=shape, mode=mode, align_corners=True)


class ModelAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
    ):
        self.model = model
        self.inputs = inputs
        self._predictions: List[Dict] = []
        self._targets: Optional[torch.Tensor] = None
        self._scores: Optional[torch.Tensor] = None
        self.model.eval()

    @property
    def predictions(self) -> List[Dict]:
        return self._predictions

    @property
    def targets(self) -> torch.Tensor:
        return self._targets

    @property
    def scores(self) -> torch.Tensor:
        return self._scores

    def forward_pass(self, class_map: Dict) -> "ModelAnalyzer":
        with torch.no_grad():
            outputs = self.model(self.inputs)

        scores, target_idxs = torch.topk(outputs, 1)
        self._predictions = [
            {
                "id": idx.item(),
                "label": class_map[str(idx.item())][1],
                "score": score.item(),
            }
            for score, idx in zip(scores, target_idxs)
        ]
        self._targets = target_idxs.squeeze()
        self._scores = scores.squeeze()

        return self  # fluent interface

    def analyze(self, attribution_config: AttributionConfig, **kwargs) -> torch.Tensor:
        if self._targets is None:
            raise RuntimeError("Must run forward_pass() before analyzing.")

        heatmap = attribution_config.attribute(
            self.model, self.inputs, self._targets, **kwargs
        )

        return heatmap

    def get_activations(
        self,
        layer: nn.Module,
        pool: bool = False,
    ) -> torch.Tensor:
        hook = self._ActivationHook()
        hook.register(layer)
        _ = self.model(self.inputs)
        hook.remove()
        return hook.activations.mean(1, keepdim=True) if pool else hook.activations

    class _ActivationHook:
        def __init__(self):
            self.activations = None
            self.hook_handle = None

        def __call__(self, module, input, output):
            self.activations = output

        def register(self, layer):
            self.hook_handle = layer.register_forward_hook(self)

        def remove(self):
            if self.hook_handle:
                self.hook_handle.remove()

    @staticmethod
    def find_layers_by_type(model: nn.Module, layer_type: type) -> List[nn.Module]:
        return [
            module
            for _, module in model.named_modules()
            if isinstance(module, layer_type)
        ]
