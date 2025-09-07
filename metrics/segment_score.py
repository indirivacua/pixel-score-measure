
import torch
import torch.nn.functional as F
from typing import Optional, List, Callable
from abc import ABC, abstractmethod

from .metrics import Metric  # base class

# ---------- Segmentation configs ----------

class SegmentConfig(ABC):
    """Abstract segmenter config.

    A SegmentConfig must implement `segment(inputs)` which returns a tensor of
    binary masks of shape (B, K, H, W). Masks must be disjoint and cover the image.
    """

    @abstractmethod
    def segment(self, inputs: torch.Tensor) -> torch.Tensor:
        """Produce disjoint segmentation masks for each image in the batch.

        Args:
            inputs: Float tensor of shape (B, C, H, W), expected in [0, 1].

        Returns:
            Tensor of shape (B, K, H, W) with {0,1} masks.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def k(self) -> int:
        """Number of segments per image."""
        raise NotImplementedError


class KmeansConfig(SegmentConfig):
    """Unsupervised image segmentation via K-means in (Lab + xy) space.

    Parameters
    ----------
    k : int
        Number of segments per image.
    use_lab : bool
        If True (default), convert RGB to CIE Lab before clustering.
        If False, cluster in linear RGB instead.
    add_xy : bool
        If True (default), append normalized (x, y) coordinates to color features.
    xy_weight : float
        Multiplicative weight for xy coordinates relative to color features.
    n_iters : int
        Number of K-means iterations.
    seed : int | None
        Random seed for centroid initialization (PyTorch RNG).
    """

    def __init__(
        self,
        k: int = 25,
        use_lab: bool = True,
        add_xy: bool = True,
        xy_weight: float = 1.0,
        n_iters: int = 10,
        seed: Optional[int] = 0,
    ):
        assert k >= 1
        self._k = int(k)
        self.use_lab = bool(use_lab)
        self.add_xy = bool(add_xy)
        self.xy_weight = float(xy_weight)
        self.n_iters = int(n_iters)
        self.seed = seed

    @property
    def k(self) -> int:
        return self._k

    @staticmethod
    def _srgb_to_linear(u: torch.Tensor) -> torch.Tensor:
        # u assumed in [0,1]
        return torch.where(
            u <= 0.04045,
            u / 12.92,
            ((u + 0.055) / 1.055).clamp(min=0) ** 2.4,
        )

    @staticmethod
    def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
        """Convert an RGB image batch in [0,1] to Lab. Shape: (B,3,H,W)."""
        B, C, H, W = rgb.shape
        assert C == 3, "RGB expected for Lab conversion"
        rgb_lin = KmeansConfig._srgb_to_linear(rgb)

        # RGB -> XYZ (D65)
        M = rgb_lin.new_tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ])  # (3,3)
        rgb_flat = rgb_lin.view(B, 3, -1)  # (B,3,N)
        xyz_flat = M @ rgb_flat  # (B,3,N)

        # Normalize by white point (D65)
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        x = xyz_flat[:, 0, :] / Xn
        y = xyz_flat[:, 1, :] / Yn
        z = xyz_flat[:, 2, :] / Zn

        eps = 216 / 24389  # ~0.008856
        kappa = 24389 / 27  # ~903.3

        def f(t):
            return torch.where(t > eps, t.pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

        fx, fy, fz = f(x), f(y), f(z)

        L = (116 * fy - 16).clamp(min=0, max=100)
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        lab = torch.stack([L, a, b], dim=1).view(B, 3, H, W)
        return lab

    @staticmethod
    def _standardize_per_image(feats: torch.Tensor) -> torch.Tensor:
        """Standardize features per image: (N, D) -> zero mean, unit std."""
        mean = feats.mean(0, keepdim=True)
        std = feats.std(0, keepdim=True).clamp_min(1e-6)
        return (feats - mean) / std

    def _build_features(self, img: torch.Tensor) -> torch.Tensor:
        """Build (N,D) features for one image (C,H,W) in [0,1]."""
        C, H, W = img.shape
        if self.use_lab:
            assert C == 3, "use_lab=True requires 3-channel RGB inputs"
            img_lab = self._rgb_to_lab(img.unsqueeze(0)).squeeze(0)  # (3,H,W)
            color = img_lab
        else:
            # cluster in linear RGB
            color = self._srgb_to_linear(img.clamp(0, 1))

        feats = [color.view(C, -1).transpose(0, 1)]  # (N,C)

        if self.add_xy:
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, H, device=img.device, dtype=img.dtype),
                torch.linspace(0, 1, W, device=img.device, dtype=img.dtype),
                indexing="ij",
            )
            xy = torch.stack([xx, yy], dim=0) * self.xy_weight  # (2,H,W)
            feats.append(xy.view(2, -1).transpose(0, 1))  # (N,2)

        feats = torch.cat(feats, dim=1)  # (N,D)
        feats = self._standardize_per_image(feats)
        return feats

    def _kmeans(self, feats: torch.Tensor) -> torch.Tensor:
        """Naive K-means in PyTorch for a single image.
        Args:
            feats: (N, D) standardized.
        Returns:
            labels: (N,) in [0, k-1].
        """
        N, D = feats.shape
        g = torch.Generator(device=feats.device)
        if self.seed is not None:
            g.manual_seed(int(self.seed))
        # Init centers by random samples (kmeans++ omitted for simplicity)
        perm = torch.randperm(N, generator=g, device=feats.device)
        centers = feats[perm[: self.k]]  # (K,D)

        for _ in range(self.n_iters):
            # Assign
            # distances (N,K): ||x - c||^2 = x^2 - 2x.c + c^2
            x2 = (feats * feats).sum(1, keepdim=True)  # (N,1)
            c2 = (centers * centers).sum(1).unsqueeze(0)  # (1,K)
            distances = x2 - 2 * feats @ centers.T + c2  # (N,K)
            labels = distances.argmin(dim=1)  # (N,)

            # Update
            new_centers = torch.zeros_like(centers)
            for k in range(self.k):
                mask = labels == k
                if mask.any():
                    new_centers[k] = feats[mask].mean(0)
                else:
                    # re-seed empty cluster randomly
                    idx = torch.randint(0, N, (1,), device=feats.device)
                    new_centers[k] = feats[idx]

            if torch.allclose(new_centers, centers, atol=1e-5, rtol=0):
                centers = new_centers
                break
            centers = new_centers

        return labels

    def segment(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.ndim == 4, "inputs must be (B,C,H,W) in [0,1]"
        B, C, H, W = inputs.shape
        device = inputs.device
        masks = torch.zeros((B, self.k, H, W), device=device, dtype=torch.float32)

        for b in range(B):
            feats = self._build_features(inputs[b])
            labels = self._kmeans(feats)  # (H*W,)
            labels = labels.view(H, W)

            # One-hot to masks
            for k in range(self.k):
                masks[b, k] = (labels == k).float()

        # Ensure a strict partition (disjoint, cover all pixels)
        # (with K-means it already is, but we make sure any ties produce 1-of-K)
        sum_masks = masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        masks = masks / sum_masks  # still binary but safe if any overlap
        masks = (masks > 0.5).float()
        return masks


# ---------- SegmentScore metric ----------

class SegmentScore(Metric):
    """Segment-wise Ablation Score (SAS).

    For each image in a batch:
      1) Segment into K disjoint regions via `seg_config` (e.g., K-means).
      2) Rank regions by mean heatmap value within each mask (ascending = least important first).
      3) Build a *cumulative* curve by progressively inserting (or deleting) regions in that order.
      4) Query the model at every step and compute the AUC of score-vs-kept-pixel-fraction.

    Compatible with per-step callbacks such as `VideoCallback`, which are invoked with
    the current binary mask for the whole batch at each step (shape (B,H,W)).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        heatmaps: torch.Tensor,
        targets: torch.Tensor,
        seg_config: SegmentConfig,
        mode: str = "deletion",  # or "insertion"
        blur_sigma: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.inputs = inputs  # (B,C,H,W) in [0,1], same device as model
        self.heatmaps = heatmaps.squeeze(1)  # (B,H,W)
        self.targets = targets
        self.seg_config = seg_config
        self.mode = mode
        self.blur_sigma = blur_sigma

        self.__dict__.update(kwargs)

        self._validate_inputs()

        # Precompute segmentation once
        with torch.no_grad():
            self.segments = self.seg_config.segment(self.inputs)  # (B,K,H,W)

        self.output_curves: Optional[torch.Tensor] = None  # (B, T, 2)
        self.blurred_inputs: Optional[torch.Tensor] = None
        if self.blur_sigma is not None:
            self._precompute_blurred_inputs()

        self._compute_importance_ordering()

    @staticmethod
    def validate_inputs(inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Batch size mismatch between inputs and targets")

    def _validate_inputs(self):
        self.validate_inputs(self.inputs, self.targets)
        if self.inputs.ndim != 4:
            raise ValueError("inputs must be 4D (B,C,H,W)")
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
        # Use a separable Gaussian via conv for speed (no torchvision dependency).
        sigma = float(self.blur_sigma)
        radius = int(max(1, 3 * sigma))
        ksize = 2 * radius + 1
        x = torch.arange(ksize, device=self.inputs.device, dtype=self.inputs.dtype) - radius
        g = torch.exp(-(x**2) / (2 * sigma * sigma))
        g = (g / g.sum()).view(1, 1, 1, -1)  # (1,1,1,K)
        pad = (radius, radius, 0, 0)
        blurred = self.inputs
        # horizontal
        blurred = F.pad(blurred, pad, mode="reflect")
        blurred = F.conv2d(blurred, g.expand(self.inputs.size(1), 1, 1, -1), groups=self.inputs.size(1))
        # vertical
        gT = g.transpose(-1, -2)  # (1,1,K,1)
        pad = (0, 0, radius, radius)
        blurred = F.pad(blurred, pad, mode="reflect")
        blurred = F.conv2d(blurred, gT.expand(self.inputs.size(1), 1, -1, 1), groups=self.inputs.size(1))
        self.blurred_inputs = blurred

    def _compute_importance_ordering(self):
        """Compute per-segment importance and ordering (ascending)."""
        B, K, H, W = self.segments.shape
        hm = self.heatmaps  # (B,H,W)
        areas = self.segments.sum(dim=(2, 3)).clamp_min(1.0)  # (B,K)

        # Mean heatmap per segment (avoid NaNs for empty segments)
        sums = (self.segments * hm.unsqueeze(1)).sum(dim=(2, 3))  # (B,K)
        means = sums / areas  # (B,K)

        # Least important first (ascending)
        self.order = torch.argsort(means, dim=1)  # (B,K)

        # Store areas (as fraction) for later
        self.segment_area_frac = (self.segments.mean(dim=(2, 3))).to(self.inputs.dtype)  # (B,K)

    def _masked_inputs_from_mask(self, current_mask: torch.Tensor) -> torch.Tensor:
        """Blend inputs with background outside current_mask (B,H,W) -> (B,C,H,W)."""
        if self.blurred_inputs is not None:
            return (
                current_mask.unsqueeze(1) * self.inputs
                + (1 - current_mask).unsqueeze(1) * self.blurred_inputs
            )
        else:
            return current_mask.unsqueeze(1) * self.inputs

    def update(
        self,
        callbacks: Optional[List[Callable]] = None,
        **kwargs,
    ):
        """Run the progressive ablation/insertion using the precomputed ordering.

        This builds self.output_curves of shape (B, T, 2), with T = K+1 steps
        including the baseline (0 kept for insertion / 1 kept for deletion).
        """
        B, C, H, W = self.inputs.shape
        K = self.segments.shape[1]
        device = self.inputs.device

        # Baseline masks
        if self.mode == "insertion":
            current_mask = torch.zeros((B, H, W), device=device, dtype=self.inputs.dtype)
        elif self.mode == "deletion":
            current_mask = torch.ones((B, H, W), device=device, dtype=self.inputs.dtype)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        T = K + 1  # include baseline
        curves = torch.zeros((B, T, 2), device=device, dtype=self.inputs.dtype)

        # Baseline model score
        with torch.no_grad():
            outputs = self.model(self._masked_inputs_from_mask(current_mask))
        scores = outputs[torch.arange(B), self.targets]
        kept_frac = current_mask.mean(dim=(1, 2))  # (B,)

        curves[:, 0, 0] = kept_frac
        curves[:, 0, 1] = scores

        if callbacks:
            for cb in callbacks:
                cb(current_mask)

        # Progressive steps
        for t in range(1, T):
            # Add (insertion) or remove (deletion) the next least-important segment
            indices = self.order[:, t - 1]  # (B,)
            # Gather the mask k for each b
            step_mask = torch.stack([self.segments[b, indices[b]] for b in range(B)], dim=0)  # (B,H,W)

            if self.mode == "insertion":
                current_mask = (current_mask + step_mask).clamp(0.0, 1.0)
            else:  # deletion
                current_mask = (current_mask - step_mask).clamp(0.0, 1.0)

            with torch.no_grad():
                outputs = self.model(self._masked_inputs_from_mask(current_mask))
            scores = outputs[torch.arange(B), self.targets]
            kept_frac = current_mask.mean(dim=(1, 2))

            curves[:, t, 0] = kept_frac
            curves[:, t, 1] = scores

            if callbacks:
                for cb in callbacks:
                    cb(current_mask)

        self.output_curves = curves

    def compute(self) -> torch.Tensor:
        if self.output_curves is None:
            raise RuntimeError("Must run update() before computing AUC.")

        # Sort by x-axis (kept fraction) for trapz stability
        sorted_indices = torch.argsort(self.output_curves[:, :, 0], dim=1)

        x = torch.gather(self.output_curves[:, :, 0], 1, sorted_indices)
        y = torch.gather(self.output_curves[:, :, 1], 1, sorted_indices)

        auc = torch.trapz(y, x, dim=1)  # (B,)
        return auc

    def reset(self):
        self.output_curves = None
