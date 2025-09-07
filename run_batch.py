import subprocess
import itertools
import glob
import os
from tqdm import tqdm
from time import perf_counter
from datetime import datetime

now = datetime.now()
date_string = now.strftime("%Y-%m-%d")

# Configuración
model_names = ["resnet101", "convnext_base", "vit_l_32"]
metric_names = ["morph", "importance", "segment"]
combinations = list(itertools.product(model_names, metric_names))

script_path = "ablation.py"
base_input = "img/imagenet_filtered"
base_output = f"/mnt/sda2/datasets/ostanchi/journal_outputs_{date_string}"

# Calculamos cuántos lotes hacen falta
all_batches = sorted(glob.glob(os.path.join(base_input, "*")))

start_time = perf_counter()

for model_name, metric_name in tqdm(combinations, desc="Combinations"):
    for input_path in all_batches:
        batch_index = input_path[-3:]
        output_path = os.path.join(base_output, model_name, f"batch_{batch_index}")

        os.makedirs(output_path, exist_ok=True)

        header = f" Model={model_name} | Filter={metric_name} | Batch={batch_index} "
        print(f"\n{header:_^80}\n", flush=True)

        cmd = [
            "/home/ostanchi/miniconda3/envs/captum/bin/python",
            script_path,
            "--model_name",
            model_name,
            "--metric_name",
            metric_name,
            "--input_path",
            input_path,
            "--output_path",
            output_path,
        ]
        subprocess.run(cmd, check=True)

print("Full Elapsed time:", perf_counter() - start_time)
