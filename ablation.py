# %%
import argparse

parser = argparse.ArgumentParser(description="Configuration")
parser.add_argument(
    "--input_path",
    type=str,
    default="img/imagenet_filtered/batch_000",
    help="Input image path",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="outputs",
    help="Output path",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="resnet101",
    help="Model name",
)
parser.add_argument(
    "--metric_name",
    type=str,
    default="morph_score",
    help="Metric name",
)
parser.add_argument(
    "--filter_option",
    type=str,
    default="erosion",
    help="Filter option (erosion or dilation)",
)
args = parser.parse_args()

# %%

########################################
######################################## Model Config
########################################

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

torch.set_printoptions(precision=4, sci_mode=False)
# torch.set_grad_enabled(False)
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)  # fmt: skip

DEVICE, DTYPE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch.float32,
)
MODEL_NAME = args.model_name
INPUT_PATH = args.input_path
OUTPUT_ROOT = args.output_path
OUTPUT_PATH = f"{OUTPUT_ROOT}/{MODEL_NAME}"
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
print(f"Model {type(model).__name__} total parameters: ", pytorch_total_params)

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
# inputs = inputs[:2]

# %%
########################################
######################################## Attributions
########################################

force_computation = False

match MODEL_NAME:
    case "vit_l_32":
        last_conv_layer = model[1].encoder.layers[-1].mlp
    case _:
        pass

from captum.attr import visualization as viz
from utils.model_analyzer import ModelAnalyzer, HeatmapUtils
from utils.attr_config import AttributionConfig
from captum.attr import Occlusion, LayerGradCam
from modules.rise import RISE
from modules.grad_cam import EnhancedLayerGradCam
import importlib, inspect

analyzer = ModelAnalyzer(model, inputs)
analyzer.forward_pass(idx_to_labels)

# conv_layers = ModelAnalyzer.find_layers_by_type(model, nn.Conv2d)
# last_conv_layer = conv_layers[-1]

rise_config = AttributionConfig(
    attribution_class=RISE,
    n_masks=4096,
    initial_mask_shapes=((7, 7),),
    blur_sigma=10.0,
    threshold=0.3,
    patience=128,
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
    attribution_class=EnhancedLayerGradCam,
    layer=last_conv_layer,
    relu_attributions=True,
    force_vit_mode="vit" in MODEL_NAME,
)

baselines = importlib.import_module("modules.baselines")
baselines_configs = [
    AttributionConfig(attribution_class=cls)
    for name, cls in inspect.getmembers(baselines, inspect.isclass)
    if cls.__module__ == "modules.baselines"
]

configs = [gradcam_config, occlusion_config, rise_config, *baselines_configs]

try:
    if force_computation:
        raise FileNotFoundError
    heatmaps = {
        str(config): torch.load(
            f"{OUTPUT_PATH}/{str(config)}.pt", map_location=DEVICE, weights_only=True
        )
        for config in configs
    }
except:
    heatmaps = {"Activations": -analyzer.get_activations(last_conv_layer, pool=True)}
    heatmaps.update({str(config): analyzer.analyze(config) for config in configs})
    [torch.save(v, f"{OUTPUT_PATH}/{str(k)}.pt") for k, v in heatmaps.items()]

heatmaps = {
    str(k): HeatmapUtils.normalize(
        HeatmapUtils.upsample(v, inputs.shape[-2:], "bicubic"), use_min=True
    )
    for k, v in heatmaps.items()
}

# %%
########################################
######################################## Score
########################################

from IPython.display import HTML
from base64 import b64encode
from utils.video import VideoCallback
from metrics.morph_score import MoprhScore

SCORE_KWARGS = {"scores": analyzer.scores, "blur_sigma": 50.0}

vc = VideoCallback(cmap="gray")

scores = {k: {} for k, v in heatmaps.items()}
for k, v in heatmaps.items():
    metric = MoprhScore(model, inputs, v, analyzer.targets, **SCORE_KWARGS)
    metric.update(callbacks=[vc])
    scores[k]["curve"], scores[k]["auc"] = metric.output_curves, metric.compute()
    vc.save_video(f"{OUTPUT_PATH_DEBUG}/{k}.mp4")
    vc.reset()
    metric.reset()

# %%
########################################
######################################## Plots
########################################

from matplotlib import ticker as tkr


def set_figsize(fig, n_rows, n_columns, width_per_column=1.2, height_per_row=1.2):
    width = n_columns * width_per_column
    height = n_rows * height_per_row
    fig.set_size_inches(width, height)


def plot_curve(ax, curve_data, auc_value):
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
    ax.xaxis.set_major_formatter(tkr.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(tkr.FormatStrFormatter("%.2f"))

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    ax.invert_xaxis()
    try:
        ax.axvline(x[-2], color="r", linestyle="--", linewidth=1)
        ax.set_xticks([x[-2], 0.0, 0.5, 1.0])
    except:
        pass

    ax.text(
        0.40,
        0.80,
        f"AUC: {auc_value:.2f}",
        transform=ax.transAxes,
        fontsize=7,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )


try:
    if force_computation:
        raise FileNotFoundError
    with open(
        f"{OUTPUT_PATH}/attributions_scores_{str(metric)}_{len(inputs)}.pkl", "rb"
    ) as f:
        data = pickle.load(f)
        inputs, attributions, predictions = (
            data["inputs"],
            data["attributions"],
            data["predictions"],
        )
except:
    attributions = [
        (h, s["curve"], s["auc"]) for h, s in zip(heatmaps.values(), scores.values())
    ]

overlay_image = True
cols = list(heatmaps.keys())

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
            )

            if row_idx == 0:
                axes[0, col_idx].set_title(f"{cols[j]}\nHeatmap", fontsize=7)
                axes[0, col_idx + 1].set_title(str(metric), fontsize=7)

    out_file = (
        f"{OUTPUT_PATH}/attributions_scores_{str(metric)}_{batch_start}_{batch_end}.jpg"
    )
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

for col, (_, _, auc) in zip(cols, attributions):
    print(
        f"{col}: {auc.mean().item()}",
        file=open(f"{OUTPUT_PATH}/auc_scores_avg_{str(metric)}_{len(inputs)}.txt", "a"),
    )

with open(
    f"{OUTPUT_PATH}/attributions_scores_{str(metric)}_{len(inputs)}.pkl", "wb"
) as f:
    pickle.dump(
        {
            "inputs": inputs,
            "attributions": attributions,
            "predictions": analyzer.predictions,
        },
        f,
    )
