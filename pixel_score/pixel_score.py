import torch
import torch.nn.functional as F
from typing import List, Callable
from typing import Optional
from .metrics import Metric
import matplotlib.pyplot as plt


class PixelScore(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        heatmaps: torch.Tensor,
        targets: torch.Tensor,
        scores: torch.Tensor,
        blur_sigma: Optional[float] = None,
    ):
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.heatmaps = heatmaps
        self.targets = targets
        self.scores = scores
        self.output_curves: Optional[torch.Tensor] = None
        self._validate_inputs()
        self.blur_sigma = blur_sigma
        self.blurred_inputs: Optional[torch.Tensor] = None
        if self.blur_sigma is not None:
            self._precompute_blurred_inputs()

    @staticmethod
    def validate_inputs(inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Batch size mismatch between inputs and targets")

    def _validate_inputs(self):
        self.validate_inputs(self.inputs, self.targets)
        if self.heatmaps.ndim != 4:
            raise ValueError("Heatmaps must be 4D tensor (B, C, H, W)")
        if self.inputs.device != self.heatmaps.device:
            raise ValueError("Inputs and heatmaps must be on the same device")

    @staticmethod
    def _calculate_kernel_size(sigma: float) -> int:
        return int(2 * torch.ceil(torch.tensor(3 * sigma)).item() + 1)

    def _precompute_blurred_inputs(self):
        from torchvision.transforms import GaussianBlur

        kernel_size = self._calculate_kernel_size(self.blur_sigma)
        blurrer = GaussianBlur(kernel_size=kernel_size, sigma=self.blur_sigma)
        self.blurred_inputs = blurrer(self.inputs).to(self.inputs.device)

    @staticmethod
    def _morphology(input: torch.Tensor, mode: str) -> torch.Tensor:
        kernel = torch.ones((1, 1, 3, 3), device=input.device, dtype=input.dtype)
        input_4d = input.unsqueeze(0).unsqueeze(0)
        conv = F.conv2d(input_4d, kernel, padding="same")

        match mode:
            case "erode":
                mask = conv == kernel.numel()
            case "dilate":
                mask = conv > 0
            case _:
                raise ValueError(f"Invalid morphology mode: {mode}")

        return mask.to(input.dtype).squeeze()

    def _batch_morphology(
        self,
        mode: str,
        target_fraction: float,
        threshold: float,
        max_iter: int,
        callbacks: List[Callable],
    ) -> torch.Tensor:
        masks = (self.heatmaps.squeeze(1) > threshold).float()
        batch_size = masks.size(0)
        result = torch.zeros(batch_size, max_iter, 2, device=masks.device)

        for b in range(batch_size):
            current_mask = masks[b].float()
            history = []

            for it in range(max_iter):
                pixel_frac = current_mask.mean()
                if self._stop_condition(mode, pixel_frac, target_fraction):
                    break

                current_mask = self._morphology(current_mask, mode)
                score = self._compute_score(b, current_mask)
                history.append((pixel_frac.item(), score))

                for callback in callbacks:
                    callback(current_mask)

            # Pad and store results
            padded_history = self._pad_history(history, max_iter, b)
            result[b] = torch.tensor(padded_history, device=masks.device)

        return result

    def _pad_history(
        self, history: List[tuple], max_length: int, batch_idx: int
    ) -> List[tuple]:
        padding = [(1.0, self.scores[batch_idx].item())] * (max_length - len(history))
        return sorted(history + padding, key=lambda x: x[0])

    def _compute_score(self, batch_idx: int, mask: torch.Tensor) -> float:
        input_4d = self.inputs[batch_idx].unsqueeze(0)
        if self.blur_sigma is not None:
            blurred_input = self.blurred_inputs[batch_idx].unsqueeze(0)
            input_masked = mask * input_4d + (1 - mask) * blurred_input
        else:
            input_masked = mask * input_4d
        with torch.no_grad():
            output = self.model(input_masked)
        return output[0, self.targets[batch_idx]].item()

    def _stop_condition(self, mode: str, current: float, target: float) -> bool:
        match mode:
            case "erode":
                return current <= target
            case "dilate":
                return current >= target
            case _:
                raise ValueError("Invalid morphology operation")

    def update(
        self,
        mode: str,
        target_fraction: float = 0.5,
        threshold: float = 0.5,
        max_iter: int = 100,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.output_curves = self._batch_morphology(
            mode, target_fraction, threshold, max_iter, callbacks
        )

    def compute(self) -> torch.Tensor:
        if self.output_curves is None:
            raise RuntimeError("Must run update() before computing AUC.")

        # Obtener y ordenar resultados
        x = torch.gather(
            self.output_curves[:, :, 0],
            1,
            torch.argsort(self.output_curves[:, :, 0], dim=1),
        )
        y = torch.gather(
            self.output_curves[:, :, 1],
            1,
            torch.argsort(self.output_curves[:, :, 0], dim=1),
        )

        # Normalización vectorizada
        y_min = y.min(dim=1, keepdim=True)[0]
        y_max = y.max(dim=1, keepdim=True)[0]
        y_range = y_max - y_min + 1e-8  # Evitar división por cero
        y_norm = (y - y_min) / y_range

        # Detectar casos constantes (ajustar tol según necesidad)
        constant_mask = y_range.squeeze() < 1e-4  # Máscara 1D

        # Cálculo de AUC vectorizado
        auc = torch.trapz(y_norm, x, dim=1)

        # Manejo de casos constantes usando el último valor normalizado
        auc[constant_mask] = y_norm[constant_mask, -1]

        return auc

    def reset(self):
        self.output_curves = None
