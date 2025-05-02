import re
import csv
from collections import defaultdict

# Ruta a tu archivo nohup
LOG_PATH = "nohup.out"
# Ruta de salida
OUTPUT_CSV = "avg_rise_iterations.csv"

# Expresiones regulares
header_re = re.compile(r"Model=(?P<model>\w+)\s*\|\s*Filter=(?P<filter>\w+)\s*\|\s*Batch=(?P<batch>\d+)")
rise_re   = re.compile(r"R I S E mask.*?(\d+)/\d+")

# Estructura para acumular: dict[(model,filter)] -> [list de max_iters por batch]
data = defaultdict(list)

with open(LOG_PATH, "r") as f:
    current_key = None
    current_max = 0

    for line in f:
        # ¿Es cabecera de un nuevo batch?
        m = header_re.search(line)
        if m:
            # Si venimos de un batch anterior, lo cerramos
            if current_key is not None:
                data[current_key].append(current_max)
            # Iniciamos uno nuevo
            model  = m.group("model")
            filt   = m.group("filter")
            current_key = (model, filt)
            current_max = 0
            continue

        # ¿Línea de RISE?
        m2 = rise_re.search(line)
        if m2 and current_key is not None:
            iter_num = int(m2.group(1))
            if iter_num > current_max:
                current_max = iter_num

    # Al final del fichero, cierra el último batch
    if current_key is not None:
        data[current_key].append(current_max)

# Ahora calculamos promedios y volcamos a CSV
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "filter", "avg_rise_iterations", "num_batches"])
    for (model, filt), iters_list in sorted(data.items()):
        avg_iters = sum(iters_list) / len(iters_list)
        writer.writerow([model, filt, f"{avg_iters:.2f}", len(iters_list)])

print(f"Resultados escritos en {OUTPUT_CSV}")
