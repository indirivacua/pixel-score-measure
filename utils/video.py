import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import write_video


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalizes a batch of tensors."""
    x = x.clone()
    for data in x:
        data[:] = (data - data.min()) / (data.max() - data.min())
    return x


class VideoCallback:
    """A callback class for generating and saving video heatmaps."""

    def __init__(self, cmap: str = "gray", fps: int = 60) -> None:
        """Initializes the callback."""
        self.cmap = cmap
        self.fps = fps
        self.heatmaps = []

    def __call__(self, heatmap: torch.Tensor) -> None:
        """Generates a heatmap from a tensor."""
        # Normalize tensor
        gray = normalize_batch(heatmap).cpu()

        # Apply the colormap and convert to uint8
        rgb = (plt.get_cmap(self.cmap)(gray)[:, :, :3] * 255).astype(np.uint8)

        # Convert back to a torch tensor and append to heatmaps
        rgb_tensor = torch.from_numpy(rgb)
        self.heatmaps.append(rgb_tensor)

    def save_video(self, path: str) -> None:
        """Saves the generated heatmaps as a video."""
        # Stack heatmaps into a tensor
        heatmaps = torch.stack(self.heatmaps, dim=0)

        # Write the video from frames (TxWxHxC)
        write_video(path, heatmaps.cpu().numpy(), fps=self.fps)

    def reset(self) -> None:
        """Resets the list of heatmaps."""
        self.heatmaps = []
