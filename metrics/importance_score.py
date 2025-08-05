import torch
import torch.nn.functional as F
from typing import List, Callable, Optional
from .metrics import Metric
from torchvision.transforms import GaussianBlur
import math


class ImportanceScore(Metric):
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

    def update(
        self,
        mode: str = "lif",
        n_steps: int = 100,
        callbacks: Optional[List[Callable]] = None,
        **kwargs,
    ):
        batch_size, H, W = self.heatmaps.shape
        total_pixels = H * W
        steps = n_steps + 1
        device = self.inputs.device

        # Coordinate grid for mapping flat indices back to 2D
        coord_grid = torch.stack(
            torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            ),
            dim=0,
        )  # (2, H, W)
        coord_flat = coord_grid.view(2, -1).permute(1, 0)  # (H*W, 2)

        heatmaps_flat = self.heatmaps.view(batch_size, -1)  # (B, H*W)

        # Sort by importance values
        if mode == "lif":
            sorted_vals, sorted_indices = torch.sort(heatmaps_flat, dim=1)  # Ascending
        elif mode == "mif":
            sorted_vals, sorted_indices = torch.sort(
                heatmaps_flat, dim=1, descending=True
            )  # Descending
        else:
            raise ValueError("mode must be 'lif' or 'mif'")

        # Compute chunk size: pixels removed per step ~ total/(steps-1)
        num_chunks = max(1, steps - 1)
        chunk_size = math.ceil(total_pixels / num_chunks)

        # Prepare randomized sorted indices, applying tie-breaking only within equal-value blocks
        rand_sorted_indices = torch.zeros_like(sorted_indices)
        for b in range(batch_size):
            vals = sorted_vals[b]
            idxs = sorted_indices[b]

            # Find boundaries of equal-value blocks
            diff = vals[1:] != vals[:-1]
            change_pts = torch.nonzero(diff, as_tuple=False).squeeze() + 1
            cp_list = (
                change_pts.tolist()
                if isinstance(change_pts.tolist(), list)
                else [int(change_pts)]
            )
            boundaries = [0] + cp_list + [total_pixels]

            reordered = []
            # Process each value block in original order
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                block = idxs[start:end]
                # Split block into subchunks of size chunk_size
                subblocks = [
                    block[i : i + chunk_size]
                    for i in range(0, block.numel(), chunk_size)
                ]
                # If multiple subblocks (i.e., block is larger than chunk_size), shuffle their order
                if len(subblocks) > 1:
                    perm = torch.randperm(len(subblocks), device=device)
                    subblocks = [subblocks[i] for i in perm]
                # Append subblocks in (possibly shuffled) order
                for sb in subblocks:
                    reordered.append(sb)

            # Concatenate all blocks
            rand_sorted_indices[b] = torch.cat(reordered)

        # Prepare fractions progression
        if mode == "lif":
            fractions = torch.linspace(1.0, 0.0, steps, device=device)
        else:
            fractions = torch.linspace(0.0, 1.0, steps, device=device)

        curves = torch.zeros((batch_size, steps, 2), device=device)

        # Main loop over fraction steps
        for step in range(steps):
            f = fractions[step].item()
            k = int(f * total_pixels)

            mask = torch.zeros((batch_size, H, W), device=device, dtype=torch.float)
            if k > 0:
                # Select k pixels in lif/mif
                if mode == "lif":
                    selected_flat_idx = rand_sorted_indices[:, -k:]
                else:
                    selected_flat_idx = rand_sorted_indices[:, :k]
                # Map to 2D and update mask
                for b in range(batch_size):
                    coords = coord_flat[selected_flat_idx[b]]
                    mask[b, coords[:, 0].long(), coords[:, 1].long()] = 1.0

            # Apply mask to inputs
            if self.blur_sigma is not None:
                masked_inputs = (
                    mask.unsqueeze(1) * self.inputs
                    + (1 - mask.unsqueeze(1)) * self.blurred_inputs
                )
            else:
                masked_inputs = mask.unsqueeze(1) * self.inputs

            # Compute model outputs and scores
            with torch.no_grad():
                outputs = self.model(masked_inputs)
            scores = outputs[torch.arange(batch_size), self.targets]

            curves[:, step, 0] = f
            curves[:, step, 1] = scores

            if callbacks:
                for callback in callbacks:
                    callback(mask)

        self.output_curves = curves

    def compute(self) -> torch.Tensor:
        if self.output_curves is None:
            raise RuntimeError("Must run update() before computing AUC.")

        curves = self.output_curves
        batch_size, steps, _ = curves.shape

        cond = (curves[:, 0, 0] == 1.0).view(-1, 1).expand(-1, steps)

        x = torch.where(
            cond,
            1 - curves[:, :, 0],  # LIF: revealed = removed fraction
            0 - curves[:, :, 0],  # MIF: revealed = preserved fraction
        )
        y = curves[:, :, 1]

        auc = torch.trapz(y, x, dim=1)
        return auc

    def reset(self):
        self.output_curves = None
