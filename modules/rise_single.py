import os
import pickle

from typing import Any, Callable, Optional, Dict, List, Union

import torch

from captum._utils.progress import progress
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution

from .rise import RISE, MaskSetConfig


class SingleRISE(Attribution):
    def __init__(self, forward_func: Callable):
        super().__init__(forward_func)
        self.forward_func = forward_func

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        n_masks: int,
        initial_mask_shapes: TensorOrTupleOfTensorsGeneric,
        mask_set_config_cls: MaskSetConfig = MaskSetConfig,
        blur_sigma: Optional[float] = None,
        patience: int = 128,
        epsilon: float = 1e-3,
        threshold: float = 0.1,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        show_progress: bool = False,
        metrics_output_path: Optional[str] = None,
        callbacks: list[Callable] = [],
    ) -> TensorOrTupleOfTensorsGeneric:
        if callbacks is None:
            callbacks = []

        rise = RISE(self.forward_func)
        all_metrics: List[Dict] = []
        all_heatmaps: List[torch.Tensor] = []

        for idx, (input_tensor, tgt) in enumerate(zip(inputs, target)):
            input_tensor = input_tensor.unsqueeze(0)
            metric: Dict = {}

            heatmap = rise.attribute(
                inputs=input_tensor,
                n_masks=n_masks,
                initial_mask_shapes=initial_mask_shapes,
                mask_set_config_cls=mask_set_config_cls,
                blur_sigma=blur_sigma,
                patience=patience,
                epsilon=epsilon,
                threshold=threshold,
                baselines=baselines,
                target=tgt,
                additional_forward_args=additional_forward_args,
                show_progress=show_progress,
                metrics=metric,
                callbacks=callbacks,
            )

            all_metrics.append(metric)
            all_heatmaps.append(heatmap)

        concatenated_heatmaps = torch.cat(all_heatmaps, dim=0)

        if metrics_output_path:
            os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
            with open(metrics_output_path, "wb") as f:
                pickle.dump(all_metrics, f)

        return concatenated_heatmaps
