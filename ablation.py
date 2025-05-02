#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import argparse

parser = argparse.ArgumentParser(description="SAM-RISE Configuration")
parser.add_argument(
    "--input_path", type=str, default="img/imagenet_filtered/", help="Input image path"
)
parser.add_argument(
    "--output_path", type=str, default="outputs_test/vit_l_32", help="Output path"
)
parser.add_argument("--model_name", type=str, default="vit_l_32", help="Model name")
parser.add_argument(
    "--filter_option",
    type=str,
    default="erosion",
    help="Filter option (erosion or dilation)",
)
args = parser.parse_args()


# In[ ]:


import torch

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.get_device_name(0))


# # Model Config

# In[ ]:


import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

import os
import json
from time import perf_counter
import random
import pickle

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


torch.set_printoptions(precision=4, sci_mode=False)


# In[ ]:


# torch.set_grad_enabled(False)


# In[ ]:


# fmt: off
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
# fmt: on


# In[ ]:


df = pd.read_csv("imagenet-nano3-1000-filtered.csv")
df["Class ID"].nunique()


# In[ ]:


DEVICE, DTYPE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch.float32,
)
MODEL_NAME = args.model_name
INPUT_PATH = args.input_path
OUTPUT_PATH = args.output_path
# OUTPUT_PATH = f"{OUTPUT_ROOT}/{MODEL_NAME}"
OUTPUT_PATH_DEBUG = f"{OUTPUT_PATH}/debug"

labels_path = "models/imagenet_class_index.json"

os.makedirs(OUTPUT_PATH_DEBUG, exist_ok=True)

model = models.get_model(MODEL_NAME, weights="DEFAULT").to(device=DEVICE, dtype=DTYPE)
model = nn.Sequential(
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    model,
    nn.Softmax(dim=1),
)
model.eval()

# Obtain the last convolutional layer
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        last_conv_layer = layer

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device=DEVICE, dtype=DTYPE)),
    ]
)

pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(f"Model {type(model).__name__} total parameters: ", pytorch_total_params)

with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)

get_class = lambda id: idx_to_labels[str(id)][0]
get_label = lambda id: idx_to_labels[str(id)][1]

load_image = lambda img_source, transform: transform(Image.open(img_source))

inputs = []
for filename in sorted(os.listdir(INPUT_PATH)):
    f = os.path.join(INPUT_PATH, filename)
    image = load_image(f, transform)
    if image.shape[0] == 1:  # do not support grayscale images
        continue
    inputs.append(image)
inputs = torch.stack(inputs, dim=0)
inputs.shape


# In[ ]:


# inputs = inputs[:2]
# inputs.shape


# In[ ]:


with torch.no_grad():
    output = model(inputs)
prediction_score, pred_label_idx = map(lambda x: x.squeeze_(), torch.topk(output, 1))

for score, label_idx in zip(prediction_score, pred_label_idx):
    id = label_idx.item()
    predicted_label = get_label(id)
    # print(f"Predicted: {predicted_label} ({id}) ({score.item():.4f})")


# In[ ]:


# plt.imshow(inputs[0].permute(1, 2, 0).detach().cpu().numpy())


# # Attributions

# In[ ]:


# model


# In[ ]:


match MODEL_NAME:
    case "vit_l_32":
        last_conv_layer = model[1].encoder.layers[-1].mlp
    case _:
        pass


# In[ ]:


force_computation = True


# In[ ]:


from utils.model_analyzer import ModelAnalyzer, HeatmapUtils
from utils.attr_config import AttributionConfig
from captum.attr import Occlusion, LayerGradCam
from modules.rise import RISE
from modules.grad_cam import EnhancedLayerGradCam

analyzer = ModelAnalyzer(model, inputs)
analyzer.forward_pass(idx_to_labels)

# conv_layers = ModelAnalyzer.find_layers_by_type(model, nn.Conv2d)
# last_conv_layer = conv_layers[-1]

rise_config = AttributionConfig(
    RISE,
    n_masks=4096,
    initial_mask_shapes=((4, 4),),
    blur_sigma=10.0,
    threshold=0.3,
    patience=64,
    epsilon=1e-3,
    show_progress=True,
)

occlusion_config = AttributionConfig(
    attribution_class=Occlusion,
    sliding_window_shapes=(3, 32, 32),
    strides=(3, 16, 16),
    baselines=0,
    callback=lambda x: torch.clamp(x, min=0)[:, :1, :],
    show_progress=True,
)
# x.abs().mean(1, keepdim=True),

gradcam_config = AttributionConfig(
    EnhancedLayerGradCam,
    layer=last_conv_layer,
    relu_attributions=True,
    force_vit_mode="vit" in MODEL_NAME,
)

configs = [rise_config, occlusion_config, gradcam_config]
names = ["heatmap_rise", "heatmap_occ", "heatmap_gc"]

try:
    if force_computation:
        raise FileNotFoundError
    heatmaps = [
        torch.load(f"{OUTPUT_PATH}/{name}.pt", map_location=DEVICE) for name in names
    ]
except:
    heatmaps = [analyzer.analyze(config) for config in configs]

[
    torch.save(heatmap, f"{OUTPUT_PATH}/{name}.pt")
    for heatmap, name in zip(heatmaps, names)
]


# In[ ]:


analyzer.predictions


# In[ ]:


# from captum.attr import visualization as viz


# In[ ]:


# _ = viz.visualize_image_attr(
#     heatmap_rise[0].permute(1, 2, 0).detach().cpu().numpy(),
#     method="heat_map",
#     sign="absolute_value",
#     cmap="jet",
#     show_colorbar=True,
# )


# In[ ]:


# _ = viz.visualize_image_attr(
#     heatmap_occ[0].permute(1, 2, 0).detach().cpu().numpy(),
#     method="heat_map",
#     sign="positive",
#     cmap="jet",
#     show_colorbar=True,
# )


# In[ ]:


# _ = viz.visualize_image_attr(
#     heatmap_gc[0].permute(1, 2, 0).detach().cpu().numpy(),
#     method="heat_map",
#     sign="positive",
#     cmap="jet",
#     show_colorbar=True,
# )


# In[ ]:


# heatmap_rise.shape, heatmap_occ.shape, heatmap_gc.shape


# In[ ]:


upsample_shape = inputs.shape[-2:]
upsample_mode = "bicubic"

heatmaps = [HeatmapUtils.upsample(h, upsample_shape, upsample_mode) for h in heatmaps]


# In[ ]:


# heatmap_rise.shape, heatmap_occ.shape, heatmap_gc.shape


# In[ ]:


# (
#     heatmap_rise.amin(dim=(1, 2, 3)),
#     heatmap_rise.amax(dim=(1, 2, 3)),
#     heatmap_occ.amin(dim=(1, 2, 3)),
#     heatmap_occ.amax(dim=(1, 2, 3)),
#     heatmap_gc.amin(dim=(1, 2, 3)),
#     heatmap_gc.amax(dim=(1, 2, 3)),
# )


# In[ ]:


heatmaps = [HeatmapUtils.normalize(h) for h in heatmaps]


# In[ ]:


# (
#     heatmap_rise.amin(dim=(1, 2, 3)),
#     heatmap_rise.amax(dim=(1, 2, 3)),
#     heatmap_occ.amin(dim=(1, 2, 3)),
#     heatmap_occ.amax(dim=(1, 2, 3)),
#     heatmap_gc.amin(dim=(1, 2, 3)),
#     heatmap_gc.amax(dim=(1, 2, 3)),
# )


# In[ ]:


heatmap_rise, heatmap_occ, heatmap_gc = heatmaps
heatmap_rise.shape, heatmap_occ.shape, heatmap_gc.shape


# In[ ]:


# plt.imshow(heatmap_rise[0].permute(1, 2, 0).detach().cpu().numpy(), cmap="jet")
# plt.colorbar()


# In[ ]:


# plt.imshow(heatmap_occ[0].permute(1, 2, 0).detach().cpu().numpy(), cmap="jet")
# plt.colorbar()


# In[ ]:


# plt.imshow(heatmap_gc[0].permute(1, 2, 0).detach().cpu().numpy(), cmap="jet")
# plt.colorbar()


# # Pixel Score

# In[ ]:


from IPython.display import HTML
from base64 import b64encode

from utils.video import VideoCallback
from pixel_score.pixel_score import PixelScore


# In[ ]:


SCORE_MODE = args.filter_option
SCORE_THRESHOLD = 0.5
SCORE_KWARGS = {"blur_sigma": 10.0}
overlay_image = True


# In[ ]:


# heatmap_rise
vc = VideoCallback(cmap="gray")
metric = PixelScore(
    model, inputs, heatmap_rise, analyzer.targets, analyzer.scores, **SCORE_KWARGS
)
metric.update(
    "erode",
    target_fraction=0.01,
    threshold=SCORE_THRESHOLD,
    max_iter=100,
    callbacks=[vc],
)
ps_erosion_curve_rise, ps_erosion_auc_rise = metric.output_curves, metric.compute()
vc.save_video(f"{OUTPUT_PATH_DEBUG}/heatmap_rise_erosion.mp4")
metric.reset()
vc.reset()
metric.update(
    "dilate",
    target_fraction=0.95,
    threshold=SCORE_THRESHOLD,
    max_iter=100,
    callbacks=[vc],
)
ps_dilation_curve_rise, ps_dilation_auc_rise = metric.output_curves, metric.compute()
vc.save_video(f"{OUTPUT_PATH_DEBUG}/heatmap_rise_dilation.mp4")


# In[ ]:


# mp4 = open(f"{OUTPUT_PATH_DEBUG}/heatmap_rise_erosion.mp4", "rb").read()
# data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

# HTML(
#     """
#     <video width=400 controls>
#         <source src="%s" type="video/mp4">
#     </video>
#     """
#     % data_url
# )


# In[ ]:


# ps_erosion_curve_rise, ps_erosion_auc_rise


# In[ ]:


# heatmap_occ
vc = VideoCallback(cmap="gray")
metric = PixelScore(
    model, inputs, heatmap_occ, analyzer.targets, analyzer.scores, **SCORE_KWARGS
)
metric.update(
    "erode",
    target_fraction=0.01,
    threshold=SCORE_THRESHOLD,
    max_iter=100,
    callbacks=[vc],
)
ps_erosion_curve_occ, ps_erosion_auc_occ = metric.output_curves, metric.compute()
vc.save_video(f"{OUTPUT_PATH_DEBUG}/heatmap_occ_erosion.mp4")
metric.reset()
vc.reset()
metric.update(
    "dilate",
    target_fraction=0.95,
    threshold=SCORE_THRESHOLD,
    max_iter=100,
    callbacks=[vc],
)
ps_dilation_curve_occ, ps_dilation_auc_occ = metric.output_curves, metric.compute()
vc.save_video(f"{OUTPUT_PATH_DEBUG}/heatmap_occ_dilation.mp4")


# In[ ]:


# mp4 = open(f"{OUTPUT_PATH_DEBUG}/heatmap_occ_erosion.mp4", "rb").read()
# data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

# HTML(
#     """
#     <video width=400 controls>
#         <source src="%s" type="video/mp4">
#     </video>
#     """
#     % data_url
# )


# In[ ]:


# ps_erosion_curve_occ, ps_erosion_auc_occ


# In[ ]:


# heatmap_gc
vc = VideoCallback(cmap="gray")
metric = PixelScore(
    model, inputs, heatmap_gc, analyzer.targets, analyzer.scores, **SCORE_KWARGS
)
metric.update(
    "erode",
    target_fraction=0.01,
    threshold=SCORE_THRESHOLD,
    max_iter=100,
    callbacks=[vc],
)
ps_erosion_curve_gc, ps_erosion_auc_gc = metric.output_curves, metric.compute()
vc.save_video(f"{OUTPUT_PATH_DEBUG}/heatmap_gc_erosion.mp4")
metric.reset()
vc.reset()
metric.update(
    "dilate",
    target_fraction=0.95,
    threshold=SCORE_THRESHOLD,
    max_iter=100,
    callbacks=[vc],
)
ps_dilation_curve_gc, ps_dilation_auc_gc = metric.output_curves, metric.compute()
vc.save_video(f"{OUTPUT_PATH_DEBUG}/heatmap_gc_dilation.mp4")


# In[ ]:


# mp4 = open(f"{OUTPUT_PATH_DEBUG}/heatmap_gc_erosion.mp4", "rb").read()
# data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

# HTML(
#     """
#     <video width=400 controls>
#         <source src="%s" type="video/mp4">
#     </video>
#     """
#     % data_url
# )


# In[ ]:


# ps_erosion_curve_gc, ps_erosion_auc_gc


# # Plots

# In[ ]:


from matplotlib import ticker as tkr


# In[ ]:


def get_score_value(mode, erosion_value, dilation_value):
    return erosion_value if mode == "erosion" else dilation_value


def set_figsize(fig, n_rows, n_columns, width_per_column=1.2, height_per_row=1.2):
    width = n_columns * width_per_column
    height = n_rows * height_per_row
    fig.set_size_inches(width, height)


def plot_curve(ax, curve_data, auc_value, is_dilation=False):
    # Convertir a numpy y ordenar
    xy = curve_data.cpu().numpy()
    sorted_indices = np.argsort(xy[:, 0])
    x = xy[sorted_indices, 0]
    y = xy[sorted_indices, 1]

    # Filtrar valores de padding (1.0) manteniendo el último punto válido
    valid_mask = x < 1.0
    first_false_index = np.argmax(~valid_mask)
    valid_mask[first_false_index] = True  # Keep 1.0 accuracy
    if np.any(valid_mask):
        x = x[valid_mask]
        y = y[valid_mask]

    ax.plot(x, y, color="tab:blue", linewidth=1)
    ax.fill_between(x, y, alpha=0.3, color="tab:blue")

    ax.tick_params(labelsize=6)
    ax.yaxis.set_major_formatter(tkr.FormatStrFormatter("%.2f"))

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    try:
        if not is_dilation:
            ax.invert_xaxis()
            ax.axvline(x[-2], color="r", linestyle="--", linewidth=1)
        else:
            ax.set_xticks([0.0, x[0], 0.5, 1.0])
            ax.xaxis.set_major_formatter(tkr.FormatStrFormatter("%.2f"))
    except IndexError:
        pass

    ax.text(
        0.40,
        0.80,
        f"AUC: {auc_value:.2f}",
        transform=ax.transAxes,
        fontsize=7,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )


# In[ ]:


attributions = [
    (
        heatmap_rise,
        get_score_value(SCORE_MODE, ps_erosion_curve_rise, ps_dilation_curve_rise),
        get_score_value(SCORE_MODE, ps_erosion_auc_rise, ps_dilation_auc_rise),
    ),
    (
        heatmap_occ,
        get_score_value(SCORE_MODE, ps_erosion_curve_occ, ps_dilation_curve_occ),
        get_score_value(SCORE_MODE, ps_erosion_auc_occ, ps_dilation_auc_occ),
    ),
    (
        heatmap_gc,
        get_score_value(SCORE_MODE, ps_erosion_curve_gc, ps_dilation_curve_gc),
        get_score_value(SCORE_MODE, ps_erosion_auc_gc, ps_dilation_auc_gc),
    ),
]


# In[ ]:


cols = ["CB-RISE (4x4)", "Occlusion", "Grad-CAM"]

sub_batch_size = 16
n_images = inputs.shape[0]
n_cols = len(attributions) * 2 + 1

for batch_start in range(0, n_images, sub_batch_size):
    batch_end = min(batch_start + sub_batch_size, n_images)
    curr_size = batch_end - batch_start

    sub_inputs = inputs[batch_start:batch_end]
    sub_attributions = [
        (
            attr[batch_start:batch_end],
            curve[batch_start:batch_end],
            auc[batch_start:batch_end],
        )
        for (attr, curve, auc) in attributions
    ]
    sub_preds = analyzer.predictions[batch_start:batch_end]

    fig, axes = plt.subplots(curr_size, n_cols, constrained_layout=True)
    set_figsize(fig, curr_size, n_cols)
    fig.set_dpi(300)

    for row_idx in range(curr_size):
        i = batch_start + row_idx
        image_np = sub_inputs[row_idx].permute(1, 2, 0).cpu().numpy()

        ax = axes[row_idx, 0]
        ax.imshow(image_np)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(sub_preds[row_idx]["label"].replace("_", " ").title(), fontsize=6)

        for j, (attr_chunk, curve_chunk, auc_chunk) in enumerate(sub_attributions):
            col_idx = j * 2 + 1

            if overlay_image:
                axes[row_idx, col_idx].imshow(image_np)

            im = axes[row_idx, col_idx].imshow(
                attr_chunk[row_idx][0].detach().cpu().numpy(),
                cmap="jet",
                alpha=0.5 if overlay_image else 1,
            )
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

            plot_curve(
                axes[row_idx, col_idx + 1],
                curve_chunk[row_idx],
                auc_chunk[row_idx].item(),
                is_dilation=(SCORE_MODE == "dilation"),
            )

            if row_idx == 0:
                axes[0, col_idx].set_title(f"{cols[j]}\nHeatmap", fontsize=7)
                axes[0, col_idx + 1].set_title(SCORE_MODE.title(), fontsize=7)

    out_file = (
        f"{OUTPUT_PATH}/attributions_scores_{SCORE_MODE}_{batch_start}_{batch_end}.jpg"
    )
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# In[ ]:


for col, (_, _, auc) in zip(cols, attributions):
    print(
        f"{col}: {auc.mean().item()}",
        file=open(f"{OUTPUT_PATH}/auc_scores_avg_{SCORE_MODE}_{len(inputs)}.txt", "a"),
    )


# In[ ]:


with open(
    f"{OUTPUT_PATH}/attributions_scores_{SCORE_MODE}_{len(inputs)}.pkl", "wb"
) as f:
    pickle.dump(
        {
            "inputs": inputs,
            "attributions": attributions,
            "predictions": analyzer.predictions,
        },
        f,
    )


# In[ ]:


# with open(
#     f"{OUTPUT_PATH}/attributions_scores_{SCORE_MODE}_{len(inputs)}.pkl", "rb"
# ) as f:
#     data = pickle.load(f)
#     inputs = data["inputs"]
#     attributions = data["attributions"]
#     predictions = data["predictions"]


# In[ ]:
