import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import write_video
from typing import List, Dict, Optional

class VideoCallback:
    """Callback para generar videos de heatmaps por elemento de batch"""
    
    def __init__(self, cmap: str = "gray", fps: int = 60):
        self.cmap = cmap
        self.fps = fps
        self.videos: Dict[int, List[torch.Tensor]] = {}
        self.batch_size: Optional[int] = None
    
    def __call__(self, heatmap_batch: torch.Tensor):
        """Procesa un batch de heatmaps"""
        if self.batch_size is None:
            self.batch_size = heatmap_batch.size(0)
            for i in range(self.batch_size):
                self.videos[i] = []
        
        heatmap_batch = heatmap_batch.cpu().float()
        
        for i in range(self.batch_size):
            hm = heatmap_batch[i]
            hm_normalized = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            
            rgb = (plt.get_cmap(self.cmap)(hm_normalized.numpy())[..., :3] * 255)
            rgb_uint8 = rgb.astype(np.uint8)
            
            self.videos[i].append(torch.from_numpy(rgb_uint8))
    
    def save_videos(self, output_dir: str, prefix: str):
        """Guarda todos los videos en el directorio especificado"""
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        
        for i, frames in self.videos.items():
            if not frames:
                dummy = torch.zeros((224, 224, 3), dtype=torch.uint8)
                frames = [dummy]
            
            video_tensor = torch.stack(frames, dim=0)
            
            video_path = os.path.join(output_dir, f"{prefix}_{i}.mp4")
            write_video(video_path, video_tensor.numpy(), fps=self.fps)
            paths.append(video_path)
        
        return paths
    
    def reset(self):
        """Reinicia el estado del callback"""
        self.videos = {}
        self.batch_size = None
