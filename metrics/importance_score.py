import torch
import torch.nn.functional as F
from typing import List, Callable, Optional
from .metrics import Metric
from torchvision.transforms import GaussianBlur


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
    ):
        batch_size, H, W = self.heatmaps.shape
        total_pixels = H * W
        steps = n_steps + 1
        device = self.inputs.device

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
        random_values = torch.rand(heatmaps_flat.shape, device=device)
        tie_breaker = 1e-6 * random_values
        heatmaps_with_tie_break = heatmaps_flat + tie_breaker

        # Ordenar por importancia
        if mode == "lif":
            sorted_vals, sorted_indices = torch.sort(
                heatmaps_with_tie_break, dim=1
            )  # Ascendente
        elif mode == "mif":
            sorted_vals, sorted_indices = torch.sort(
                heatmaps_with_tie_break, dim=1, descending=True
            )  # Descendente
        else:
            raise ValueError("mode must be 'lif' or 'mif'")

        # Set fraction progression based on mode
        if mode == "lif":
            fractions = torch.linspace(
                1.0, 0.0, steps, device=device
            )  # Start at 1.0 (all), end at 0.0 (none)
        elif mode == "mif":
            fractions = torch.linspace(
                0.0, 1.0, steps, device=device
            )  # Start at 0.0 (none), end at 1.0 (all)
        else:
            raise ValueError("mode must be 'lif' or 'mif'")
        curves = torch.zeros((batch_size, steps, 2), device=device)

        for step in range(steps):
            f = fractions[step].item()
            k = int(f * total_pixels)  # Número de píxeles a conservar

            # Crear máscara vacía
            mask = torch.zeros((batch_size, H, W), device=device, dtype=torch.float)

            if k > 0:
                # Obtener índices de los píxeles a conservar
                if mode == "lif":
                    # Conservar los píxeles más importantes (últimos k)
                    selected_flat_idx = sorted_indices[:, -k:]  # (B, k)
                elif mode == "mif":
                    # Conservar los píxeles más importantes (primeros k)
                    selected_flat_idx = sorted_indices[:, :k]  # (B, k)
                else:
                    raise ValueError("mode must be 'lif' or 'mif'")

                # Convertir índices planos a coordenadas espaciales
                for b in range(batch_size):
                    # Obtener coordenadas para los índices seleccionados
                    selected_coords = coord_flat[selected_flat_idx[b]]  # (k, 2)
                    i_coords = selected_coords[:, 0].long()
                    j_coords = selected_coords[:, 1].long()

                    # Activar píxeles en las coordenadas
                    mask[b, i_coords, j_coords] = 1.0

            if self.blur_sigma is not None:
                masked_inputs = (
                    mask.unsqueeze(1) * self.inputs
                    + (1 - mask.unsqueeze(1)) * self.blurred_inputs
                )
            else:
                masked_inputs = mask.unsqueeze(1) * self.inputs

            with torch.no_grad():
                outputs = self.model(masked_inputs)
            scores = outputs[torch.arange(batch_size), self.targets]  # (B,)

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
