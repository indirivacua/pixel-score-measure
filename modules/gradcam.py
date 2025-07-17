import math
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from captum._utils.common import (
    _format_tensor_into_tuples,
    _format_additional_forward_args,
)
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum.attr import LayerGradCam


class EnhancedLayerGradCam(LayerGradCam):
    """
    GradCAM implementation adapted for Vision Transformers (ViT) and standard CNNs.
    - In ViT mode, excludes the CLS token and reconstructs spatial heatmaps.
    - Otherwise, falls back to standard LayerGradCam.
    """

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: Optional[Union[int, Tuple[int, ...]]] = None,
        additional_forward_args: Optional[Any] = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
        attr_dim_summation: bool = True,
        force_vit_mode: bool = False,
    ) -> Tensor:
        """
        Args:
            force_vit_mode: if False, uses standard LayerGradCam.attribute;
                            if True, computes ViT-specific GradCAM.
        """
        # Standard CNN behaviour
        if not force_vit_mode:
            return super().attribute(
                inputs,
                target,
                additional_forward_args,
                attribute_to_layer_input,
                relu_attributions,
                attr_dim_summation,
            )

        # --- ViT mode ---
        # 1) Format inputs
        inputs_tuple = _format_tensor_into_tuples(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        # 2) Compute gradients and activations (without passing grad_kwargs)
        layer_grads, layer_evals = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs_tuple,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        # Extract first tuple element
        grad = layer_grads[0]  # (batch, seq_len, D)
        feat = layer_evals[0]  # (batch, seq_len, D)

        # 3) Compute channel weights
        weights = grad.mean(dim=1)  # (batch, D)

        # 4) Compute per-token GradCAM
        cam_tokens = (feat * weights.unsqueeze(1)).sum(dim=2)  # (batch, seq_len)

        # 5) Drop CLS token and reshape
        cam_tokens = cam_tokens[:, 1:]
        batch, num_patches = cam_tokens.shape
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"Number of patches ({num_patches}) is not a perfect square."
            )
        cam_grid = cam_tokens.reshape(batch, grid_size, grid_size)

        # 6) Optional ReLU and normalization
        if relu_attributions:
            cam_grid = torch.relu(cam_grid)
        cam_min = cam_grid.view(batch, -1).amin(dim=1)[:, None, None]
        cam_max = cam_grid.view(batch, -1).amax(dim=1)[:, None, None]
        heatmap = (cam_grid - cam_min) / (cam_max - cam_min + 1e-8)

        # 7) Return shape (batch,1,grid,grid)
        return heatmap.unsqueeze(1)
