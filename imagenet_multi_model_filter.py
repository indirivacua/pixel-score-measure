import json
import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm.autonotebook import tqdm

# Initial Setup
torch.set_printoptions(precision=4, sci_mode=False)
torch.set_grad_enabled(False)

# Configuration
model_names = ["vgg19", "convnext_base", "vit_l_32"]
rootdb_path = "/mnt/sda2/datasets/imagenet1k"
image_paths = f"{rootdb_path}/imagenet-nano3/val/image_paths.csv"
output_path = "imagenet-nano3-1000-filtered.csv"
labels_path = "models/imagenet_class_index.json"
destination = "img/imagenet_filtered"
batch_size = 256
score_confidence = 0.7

DEVICE, DTYPE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch.float32,
)

# Load models
model_list = []
for model_name in model_names:
    model = models.get_model(model_name, weights="DEFAULT").to(
        device=DEVICE, dtype=DTYPE
    )
    model = nn.Sequential(
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        model,
        nn.Softmax(dim=1),
    )
    model.eval()
    model_list.append(model)

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device=DEVICE, dtype=DTYPE)),
    ]
)

# Load labels
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)

# Helper functions
get_class = lambda id: idx_to_labels[str(id)][0]
get_label = lambda id: idx_to_labels[str(id)][1]

load_image = lambda img_source, transform: transform(Image.open(img_source))


def process_image(img_source, transform):
    image = load_image(img_source, transform)
    image = (
        image.expand(3, -1, -1)
        if image.shape[0] == 1  # Grayscale image
        else image[:3] if image.shape[0] > 3 else image  # RGBA image
    )
    return image


# Process data in batches
df = pd.read_csv(image_paths)
df["Image Path"] = (
    rootdb_path + "/" + df["Image Path"].str.replace("\\", "/", regex=False)
)

df_output = []

for batch_start in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    batch_end = min(batch_start + batch_size, len(df))
    batch_df = df.iloc[batch_start:batch_end]

    # Load batch images
    batch_images = []
    batch_classes = []
    for _, row in batch_df.iterrows():
        try:
            image = process_image(row["Image Path"], transform)
            batch_images.append(image)
            batch_classes.append(row["Class"])
        except Exception as e:
            print(f"Error loading {row['Image Path']}: {e}")
            continue

    if not batch_images:
        continue

    batch_tensor = torch.stack(batch_images).to(DEVICE, dtype=DTYPE)

    # Process batch through models
    batch_results = []
    for model in model_list:
        with torch.no_grad():
            output = model(batch_tensor)
            prediction_score, pred_label_idx = torch.topk(output, 1)
            batch_results.append(
                (prediction_score.squeeze(1), pred_label_idx.squeeze(1))
            )

    # Process batch results
    for i, (_, row) in enumerate(batch_df.iterrows()):
        if i >= len(batch_images):  # In case some images failed to load
            continue

        true_class = batch_classes[i]
        model_inp_res = []

        for j, _ in enumerate(model_list):
            score, id = batch_results[j][0][i].item(), batch_results[j][1][i].item()
            classp = get_class(id)
            labelp = get_label(id)
            model_inp_res.append((score, id, classp, labelp))

        if all(
            score >= score_confidence and classp == true_class
            for score, _, classp, _ in model_inp_res
        ):
            df_row = {
                "Image Path": row["Image Path"],
                "Class": true_class,
                "Class ID": model_inp_res[0][1],  # Using first model's prediction ID
                "Class Label": model_inp_res[0][
                    3
                ],  # Using first model's prediction label
            }
            for model_name, res in zip(model_names, model_inp_res):
                df_row[f"Score {model_name}"] = res[0]
            df_output.append(df_row)

# Save results
df_output = pd.DataFrame(df_output)
df_output.to_csv(output_path, index=False)

# Optional image saving in batches
save_inputs = input("Do you want to save the images? (Y/N): ")
if save_inputs.upper() == "Y":
    for batch_start in tqdm(
        range(0, len(df_output), batch_size), desc="Saving batches"
    ):
        batch_end = min(batch_start + batch_size, len(df_output))
        batch_df = df_output.iloc[batch_start:batch_end]
        batch_idx = batch_start // batch_size
        batch_folder = os.path.join(destination, f"batch_{batch_idx:03d}")
        os.makedirs(batch_folder, exist_ok=True)
        for _, row in batch_df.iterrows():
            src = row["Image Path"]
            if os.path.exists(src):
                dst = os.path.join(batch_folder, os.path.basename(src))
                shutil.copy(src, dst)
            else:
                print(f"The image {src} does not exist.")
