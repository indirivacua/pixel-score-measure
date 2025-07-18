# %%

import os
import glob
import csv

# Directorio base que contiene las carpetas de cada modelo
BASE_DIR = (
    "/mnt/sda2/datasets/ostanchi/cacic_outputs_2025-07-16"  # ajústalo a tu ruta
)

# Métodos que aparecen en los archivos
METHODS = ["Activations", "EnhancedLayerGradCam", "Occlusion", "RISE", "CentralAttribution", "NormalAttribution", "OnePixelAttribution", "UniformAttribution"]
# Filtros que tienes
FILTERS = ["MorphScore", "ImportanceScore"]

# Lista donde iremos acumulando filas para el CSV resultado
rows = []

for model_name in sorted(os.listdir(BASE_DIR)):
    model_dir = os.path.join(BASE_DIR, model_name)
    if not os.path.isdir(model_dir):
        continue

    for filt in FILTERS:
        # Patrón para localizar todos los archivos de este filtro en todos los batches
        pattern = os.path.join(model_dir, "batch_*", f"attributions_scores_{filt}_64.txt")
        files = glob.glob(pattern)
        if not files:
            continue

        # Acumuladores de sumas y contadores por método
        sums = {m: 0.0 for m in METHODS}
        counts = {m: 0 for m in METHODS}

        # Leer cada archivo y sumar valores
        for fn in files:
            with open(fn, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Esperamos formato: "Método: valor"
                    try:
                        method, val = line.split(":", 1)
                        method = method.strip()
                        val = float(val.strip())
                        if method in sums:
                            sums[method] += val
                            counts[method] += 1
                    except ValueError:
                        # línea inesperada, la saltamos
                        continue

        # Calcular promedio para cada método
        for method in METHODS:
            if counts[method] > 0:
                avg = sums[method] / counts[method]
            else:
                avg = float("nan")
            rows.append([model_name, filt, method, avg])

# Escribir CSV de salida
OUT_CSV = "avg_auc_scores_by_model_and_filter.csv"
with open(OUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "filter", "method", "avg_auc_score"])
    writer.writerows(rows)

print(f"Promedios escritos en {OUT_CSV}")

# %%

import pandas as pd
import matplotlib.pyplot as plt

# Configuration
csv_path = "avg_auc_scores_by_model_and_filter.csv"  # Update this path if needed
output_path = "avg_auc_scores_by_model_and_filter.jpg"
dpi = 300
bar_width = 0.2
figsize = (12, 6)

# Read data
df = pd.read_csv(csv_path)

# Extract unique values
models = df["model"].unique()
methods = df["method"].unique()
filters = df["filter"].unique()

# Create subplots for each filter type
fig, axes = plt.subplots(1, len(filters), figsize=figsize, sharey=True)
if len(filters) == 1:
    axes = [axes]

for ax, filter_type in zip(axes, filters):
    subset = df[df["filter"] == filter_type]
    x_positions = range(len(models))
    for i, method in enumerate(methods):
        auc_scores = subset[subset["method"] == method]["avg_auc_score"].values
        ax.bar(
            [x + i * bar_width for x in x_positions],
            auc_scores,
            width=bar_width,
            label=method,
        )
    ax.set_title(f"{filter_type}")#.capitalize()}")
    ax.set_xticks([x + bar_width for x in x_positions])
    # ax.set_xticklabels(
    #     [m for m in models], rotation=45, ha="center"
    # )  # m.replace('_', ' ').title()
    # ax.set_xlabel("Model")
    ax.set_xticklabels([])
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

axes[0].set_ylabel("Average AUC Score")
# fig.suptitle("Average Erosion and Dilation AUC Scores by Model and Method")
fig.suptitle("Average AUC Scores")
fig.legend(methods, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 0.05))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure without rendering
fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

# %%
