import torch
import torch.nn.functional as F
from typing import List, Callable
from typing import Optional
from .metrics import Metric
from torchvision.transforms import GaussianBlur


class MorphScore(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        heatmaps: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.heatmaps = heatmaps.squeeze(1)  # (B, H, W)
        self.targets = targets
        self.output_curves: Optional[torch.Tensor] = None

        self.__dict__.update(kwargs)
        self._validate_inputs()
        self.blurred_inputs: Optional[torch.Tensor] = None
        if self.blur_sigma is not None:
            self._precompute_blurred_inputs()

    @staticmethod
    def validate_inputs(inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Batch size mismatch between inputs and targets")

    def _validate_inputs(self):
        self.validate_inputs(self.inputs, self.targets)
        if self.heatmaps.ndim != 3:
            raise ValueError("Heatmaps must be 3D tensor (B, H, W)")
        if self.inputs.shape[0] != self.heatmaps.shape[0]:
            raise ValueError("Batch size mismatch between inputs and heatmaps")
        if self.inputs.device != self.heatmaps.device:
            raise ValueError("Inputs and heatmaps must be on the same device")

    @staticmethod
    def _calculate_kernel_size(sigma: float) -> int:
        return int(2 * torch.ceil(torch.tensor(3 * sigma)).item() + 1)

    def _precompute_blurred_inputs(self):
        kernel_size = self._calculate_kernel_size(self.blur_sigma)
        blurrer = GaussianBlur(kernel_size=kernel_size, sigma=self.blur_sigma)
        self.blurred_inputs = blurrer(self.inputs).to(self.inputs.device)

    @staticmethod
    def _batch_morphology_step(input: torch.Tensor, mode: str) -> torch.Tensor:
        kernel = torch.ones((1, 1, 3, 3), device=input.device, dtype=input.dtype)
        input_4d = input.unsqueeze(1)  # (B, 1, H, W)
        conv = F.conv2d(input_4d, kernel, padding="same")

        match mode:
            case "erode":
                mask = conv == kernel.numel()
            case "dilate":
                mask = conv > 0
            case _:
                raise ValueError(f"Invalid morphology mode: {mode}")

        return mask.to(input.dtype).squeeze(1)  # (B, H, W)

    def _batch_morphology(
        self,
        mode: str,
        threshold: float,
        target_fraction: float,
        n_steps: int,
        callbacks: List[Callable],
    ) -> torch.Tensor:
        masks = (self.heatmaps > threshold).float()
        batch_size, H, W = masks.shape
        device = masks.device

        curves = torch.zeros((batch_size, n_steps, 2), device=device)

        if not hasattr(self, 'scores') or self.scores is None:
            with torch.no_grad():
                original_outputs = self.model(self.inputs)
            self.scores = original_outputs[torch.arange(batch_size), self.targets]

        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        current_masks = masks.clone()
        last_active_step = torch.zeros(batch_size, dtype=torch.long, device=device) - 1

        for step in range(n_steps):
            pixel_fracs = current_masks.mean(dim=(1, 2))  # (B,)

            if self.blur_sigma is not None:
                masked_inputs = (
                    current_masks.unsqueeze(1) * self.inputs
                    + (1 - current_masks.unsqueeze(1)) * self.blurred_inputs
                )
            else:
                masked_inputs = current_masks.unsqueeze(1) * self.inputs

            with torch.no_grad():
                outputs = self.model(masked_inputs)
            scores = outputs[torch.arange(batch_size), self.targets]  # (B,)

            curves[:, step, 0] = pixel_fracs
            curves[:, step, 1] = scores

            last_active_step[active] = step

            if callbacks:
                for callback in callbacks:
                    callback(current_masks)

            if mode == "erode":
                still_active = pixel_fracs > target_fraction
            elif mode == "dilate":
                still_active = pixel_fracs < target_fraction

            active = active & still_active

            if not active.any():
                # if step < n_steps - 1:
                #     curves[:, step + 1 :, 0] = pixel_fracs.unsqueeze(1)
                #     curves[:, step + 1 :, 1] = scores.unsqueeze(1)
                break

            if active.any():
                active_masks = current_masks[active]
                morphed_masks = self._batch_morphology_step(active_masks, mode)
                current_masks[active] = morphed_masks

        for i in range(batch_size):
            start_fill = last_active_step[i] + 1
            if start_fill < n_steps:
                curves[i, start_fill:, 0] = 1.0
                curves[i, start_fill:, 1] = self.scores[i]

        return curves

    def update(
        self,
        mode: str = "erode",
        threshold: float = 0.5,
        target_fraction: float = 0.01,
        n_steps: int = 100,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.output_curves = self._batch_morphology(
            mode, threshold, target_fraction, n_steps, callbacks
        )

    def compute(self) -> torch.Tensor:
        if self.output_curves is None:
            raise RuntimeError("Must run update() before computing AUC.")

        sorted_indices = torch.argsort(self.output_curves[:, :, 0], dim=1)

        x = torch.gather(
            self.output_curves[:, :, 0],
            1,
            sorted_indices,
        )
        y = torch.gather(
            self.output_curves[:, :, 1],
            1,
            sorted_indices,
        )

        auc = torch.trapz(y, x, dim=1)
        return auc

    def reset(self):
        self.output_curves = None

    def __str__(self):
        return "MES"