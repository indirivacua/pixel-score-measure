import os
import glob
import csv

# Directorio base que contiene las carpetas de cada modelo
BASE_DIR = "/mnt/sda2/datasets/ostanchi/journal_outputs_2025-05-01"  # ajústalo a tu ruta

# Métodos que aparecen en los archivos
METHODS = ["CB-RISE (4x4)", "Occlusion", "Grad-CAM"]
# Filtros que tienes
FILTERS = ["erosion", "dilation"]

# Lista donde iremos acumulando filas para el CSV resultado
rows = []

for model_name in sorted(os.listdir(BASE_DIR)):
    model_dir = os.path.join(BASE_DIR, model_name)
    if not os.path.isdir(model_dir):
        continue

    for filt in FILTERS:
        # Patrón para localizar todos los archivos de este filtro en todos los batches
        pattern = os.path.join(model_dir, "batch_*", f"auc_scores_avg_{filt}_64.txt")
        files = glob.glob(pattern)
        if not files:
            continue

        # Acumuladores de sumas y contadores por método
        sums = {m: 0.0 for m in METHODS}
        counts = {m: 0   for m in METHODS}

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
                            sums[method]  += val
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
