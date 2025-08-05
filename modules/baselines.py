import math
import torch
from captum.attr import Attribution
from typing import Any, Callable, Optional, Tuple, Union
from torch import Tensor


class OnePixelAttribution(Attribution):
    def __init__(self, forward_func: Callable):
        """
        Atribución concentrada en un solo píxel (entropía mínima).

        Args:
            forward_func (Callable): Función de forward del modelo (no se usa en este método)
        """
        super().__init__(forward_func)

    def attribute(
        self,
        inputs: Tensor,
        baselines: Optional[Tensor] = None,
        target: Optional[int] = None,
        additional_forward_args: Any = None,
        **kwargs
    ) -> Tensor:
        """
        Genera heatmaps con un único píxel activo (valor 1.0) seleccionado aleatoriamente.

        Args:
            inputs (Tensor): Tensor de entrada con forma (B, C, H, W)
            **kwargs: Argumentos adicionales (ignorados)

        Returns:
            Tensor: Heatmaps con forma (B, 1, H, W)
        """
        B, C, H, W = inputs.shape

        # Crear heatmaps de ceros
        heatmaps = torch.zeros(B, 1, H, W, device=inputs.device, dtype=inputs.dtype)

        # Seleccionar píxeles aleatorios para cada elemento del batch
        for i in range(B):
            h_idx = torch.randint(0, H, (1,))
            w_idx = torch.randint(0, W, (1,))
            heatmaps[i, 0, h_idx, w_idx] = 1.0

        return heatmaps

    def __str__(self):
        return r"$\mathbf{1}$-pixel"


class UniformAttribution(Attribution):
    def __init__(self, forward_func: Callable):
        """
        Atribución uniforme en todos los píxeles (entropía máxima).

        Args:
            forward_func (Callable): Función de forward del modelo
            k (float, optional): Valor constante para todos los píxeles.
                                 Si es None, se usa un valor aleatorio por heatmap.
        """
        super().__init__(forward_func)

    def attribute(
        self,
        inputs: Tensor,
        k: Optional[float] = None,
        baselines: Optional[Tensor] = None,
        target: Optional[int] = None,
        additional_forward_args: Any = None,
        **kwargs
    ) -> Tensor:
        """
        Genera heatmaps con valores uniformes.

        Args:
            inputs (Tensor): Tensor de entrada con forma (B, C, H, W)
            **kwargs: Argumentos adicionales (ignorados)

        Returns:
            Tensor: Heatmaps con forma (B, 1, H, W)
        """
        B, C, H, W = inputs.shape

        if k is not None:
            # Usar valor constante k para todos los píxeles
            heatmaps = torch.full(
                (B, 1, H, W), k, device=inputs.device, dtype=inputs.dtype
            )
        else:
            # Generar un valor aleatorio diferente para cada elemento del batch
            k_values = torch.rand(B, 1, 1, 1, device=inputs.device, dtype=inputs.dtype)
            heatmaps = k_values.expand(B, 1, H, W)

        # Normalizar a distribución de probabilidad
        # heatmaps = heatmaps / heatmaps.sum(dim=(2, 3), keepdim=True)

        return heatmaps

    def __str__(self):
        return r"$\mathcal{U}$-pixel"


class NormalAttribution(Attribution):
    def __init__(self, forward_func: Callable):
        """
        Atribución con distribución normal (entropía intermedia).

        Args:
            forward_func (Callable): Función de forward del modelo
            mean (float): Media de la distribución normal
            std (float): Desviación estándar de la distribución normal
        """
        super().__init__(forward_func)

    def attribute(
        self,
        inputs: Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        baselines: Optional[Tensor] = None,
        target: Optional[int] = None,
        additional_forward_args: Any = None,
        **kwargs
    ) -> Tensor:
        """
        Genera heatmaps con valores muestreados de una distribución normal.

        Args:
            inputs (Tensor): Tensor de entrada con forma (B, C, H, W)
            **kwargs: Argumentos adicionales (ignorados)

        Returns:
            Tensor: Heatmaps con forma (B, 1, H, W)
        """
        B, C, H, W = inputs.shape

        # Generar valores aleatorios con distribución normal
        heatmaps = torch.randn(B, 1, H, W, device=inputs.device, dtype=inputs.dtype)
        heatmaps = heatmaps * std + mean

        # Convertir a valores no negativos (absolutos)
        heatmaps = torch.abs(heatmaps)

        # Normalizar a distribución de probabilidad
        heatmaps = heatmaps / heatmaps.sum(dim=(2, 3), keepdim=True)

        return heatmaps

    def __str__(self):
        return r"$\mathcal{N}$-pixel"


class CentralAttribution(Attribution):
    def __init__(self, forward_func: Callable):
        """
        Atribución que activa un cuadrado central de píxeles, cubriendo un porcentaje dado del total.

        Args:
            forward_func (Callable): Función de forward del modelo
            percentage (float): Fracción de píxeles totales que debe cubrir el cuadrado central (entre 0 y 1)
        """
        super().__init__(forward_func)

    def attribute(
        self,
        inputs: Tensor,
        percentage: float = 0.1,
        baselines: Optional[Tensor] = None,
        target: Optional[int] = None,
        additional_forward_args: Any = None,
        **kwargs
    ) -> Tensor:
        """
        Genera heatmaps con una región cuadrada central activa.

        Args:
            inputs (Tensor): Tensor de entrada con forma (B, C, H, W)
            **kwargs: Argumentos adicionales (ignorados)

        Returns:
            Tensor: Heatmaps con forma (B, 1, H, W)
        """
        B, C, H, W = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        # Manejar caso de porcentaje cero
        if percentage <= 0:
            return torch.zeros(B, 1, H, W, device=device, dtype=dtype)

        # Calcular número total de píxeles y tamaño del cuadrado central
        total_pixels = H * W
        k = percentage * total_pixels

        # Calcular lado del cuadrado (redondeado al entero más cercano)
        s = round(math.sqrt(k))
        s = int(max(1, min(s, min(H, W))))  # Asegurar tamaño válido [1, min(H,W)]

        # Calcular coordenadas del cuadrado central
        top = (H - s) // 2
        left = (W - s) // 2

        # Crear heatmap con cuadrado central
        heatmap = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        heatmap[:, :, top : top + s, left : left + s] = 1.0

        # Normalizar para formar distribución de probabilidad
        heatmap = heatmap / (s * s)

        return heatmap

    def __str__(self):
        return r"$\mathbf{\%}$-pixel"
