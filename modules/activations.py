from captum.attr import Attribution
from typing import Any, Callable, Optional
import torch
import torch.nn as nn


class Activations(Attribution):
    def __init__(self, forward_func: Callable, layer: nn.Module):
        """
        Attribution method that extracts activations from a specific layer.

        Args:
            forward_func (Callable): Model forward function
            layer (nn.Module): Target layer to extract activations from
        """
        super().__init__(forward_func)
        self.layer = layer
        self.hook = self._ActivationHook()

    class _ActivationHook:
        def __init__(self):
            self.activations = None
            self.hook_handle = None

        def __call__(self, module, input, output):
            self.activations = output.detach()

        def register(self, layer):
            if self.hook_handle:
                self.hook_handle.remove()
            self.hook_handle = layer.register_forward_hook(self)

        def remove(self):
            if self.hook_handle:
                self.hook_handle.remove()
                self.hook_handle = None

    def attribute(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        average_across_channels: bool = False,
        additional_forward_args: Any = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Extracts activations from the specified layer.

        Args:
            inputs (Tensor): Input tensor (B, C, H, W)
            average_across_channels (bool): Whether to average activations across channels
            **kwargs: Additional arguments (ignored)

        Returns:
            Tensor: Activations from the target layer (B, C', H', W') or (B, 1, H', W') if averaged
        """
        self.hook.register(self.layer)

        with torch.no_grad():
            if additional_forward_args is not None:
                _ = self.forward_func(inputs, *additional_forward_args)
            else:
                _ = self.forward_func(inputs)

        self.hook.remove()
        activations = self.hook.activations

        if average_across_channels and activations.ndim == 4:
            return activations.mean(dim=1, keepdim=True)
        return activations
