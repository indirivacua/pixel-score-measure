import os
import shutil

# Ruta raíz donde están las carpetas batch
root = "/mnt/sda2/datasets/ostanchi/cacic_outputs_2025-07-16/resnet101"

# Iteramos por los batch
for batch in os.listdir(root):
    batch_path = os.path.join(root, batch)
    nested_path = os.path.join(batch_path, "resnet101")

    # Verificamos si existe la subcarpeta resnet101 dentro del batch
    if os.path.isdir(nested_path):
        for file_name in os.listdir(nested_path):
            src = os.path.join(nested_path, file_name)
            dst = os.path.join(batch_path, file_name)
            shutil.move(src, dst)  # Movemos el archivo a la carpeta batch
        os.rmdir(nested_path)  # Eliminamos la subcarpeta vacía

print("¡Listo! Se reorganizaron los archivos correctamente.")
