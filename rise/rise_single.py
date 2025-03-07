from captum.attr import Attribution
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric

from typing import Callable, Optional, Dict

from .rise import RISE

import torch
import pickle


class SingleRISE(Attribution):
    def __init__(self, forward_func: Callable):
        super().__init__(forward_func)
        self.forward_func = forward_func

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        n_masks: int,
        initial_mask_shapes: TensorOrTupleOfTensorsGeneric,
        # mask_set_config_cls: MaskSetConfig = MaskSetConfig,
        blur_sigma: float = None,
        patience: int = 128,
        d_epsilon: float = 1e-3,
        threshold: float = 0.1,
        # baselines: BaselineType = None,
        target: TargetType = None,
        # additional_forward_args: Any = None,
        show_progress: bool = False,
        metrics: dict = None,
        metrics_name: str = None,
        callbacks: list[Callable] = [],
    ) -> TensorOrTupleOfTensorsGeneric:
        rise = RISE(self.forward_func)
        metrics = []
        heatmaps = []
        for input, tgt in zip(inputs, target):
            input = input.unsqueeze(dim=0)
            metric = {}
            heatmap = rise.attribute(
                input,
                n_masks=n_masks,
                initial_mask_shapes=initial_mask_shapes,
                blur_sigma=blur_sigma,
                target=tgt,
                patience=patience,
                d_epsilon=d_epsilon,
                threshold=threshold,
                show_progress=show_progress,
                metrics=metric,
            )
            metrics.append(metric)
            heatmaps.append(heatmap)
        heatmaps = torch.cat(heatmaps, dim=0)
        with open(metrics_name, "wb") as f:
            pickle.dump(metrics, f)
        return heatmaps
