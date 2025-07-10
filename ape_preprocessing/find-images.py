import os
import random
import numpy as np
from PIL import Image
import shutil

# 1. Definisi kelas dalam bentuk index
CLASS_INDEX_DEFINITIONS = {
    1: 'built-up',
    2: 'farmland',
    3: 'forest',
    4: 'meadow',
    5: 'water'
}

# 2. Parameter
INPUT_LABEL_DIR = 'Annotation__index'          # Folder label index
INPUT_IMAGE_16BIT_DIR = 'image_16bit'          # Folder citra 16-bit (.tiff)
INPUT_IMAGE_8BIT_DIR = 'image_8bit'            # Folder citra 8-bit (.tif)

OUTPUT_LABEL_DIR = 'selected_label_index'
OUTPUT_IMAGE_16BIT_DIR = 'selected_image_16bit'
OUTPUT_IMAGE_8BIT_DIR = 'selected_image_8bit'

NUM_TO_SELECT = 15  # Jumlah gambar yang akan dipilih

def main():
    valid_label_paths = []

    print("Memulai proses pemilihan gambar (berbasis label index)...")

    for filename in os.listdir(INPUT_LABEL_DIR):
        if not filename.lower().endswith(('.png', '.tif')):
            continue

        filepath = os.path.join(INPUT_LABEL_DIR, filename)

        try:
            img = Image.open(filepath)
            label_array = np.array(img)
            unique_labels = np.unique(label_array)

            contains_all_classes = all(label in unique_labels for label in CLASS_INDEX_DEFINITIONS.keys())

            if contains_all_classes:
                valid_label_paths.append(filepath)
                print(f"Ditemukan gambar valid: {filename}")

        except Exception as e:
            print(f"Gagal memproses {filename}: {str(e)}")

    print(f"\nTotal gambar yang mengandung semua kelas: {len(valid_label_paths)}")

    if len(valid_label_paths) < NUM_TO_SELECT:
        print(f"Peringatan: Hanya ada {len(valid_label_paths)} gambar valid, memilih semuanya")
        selected_labels = valid_label_paths
    else:
        selected_labels = random.sample(valid_label_paths, NUM_TO_SELECT)

    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_16BIT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_8BIT_DIR, exist_ok=True)

    print("\nMenyalin gambar dan label:")
    for i, label_path in enumerate(selected_labels, 1):
        label_filename = os.path.basename(label_path)
        base_name = label_filename.replace('_5label.png', '').replace('_5label.tif', '')  # nama dasar tanpa label

        # 1. Salin label
        label_output_path = os.path.join(OUTPUT_LABEL_DIR, label_filename)
        shutil.copy2(label_path, label_output_path)

        # 2. Salin citra 16bit (.tiff)
        image_16bit_filename = f"{base_name}.tiff"
        image_16bit_path = os.path.join(INPUT_IMAGE_16BIT_DIR, image_16bit_filename)
        if os.path.exists(image_16bit_path):
            output_16bit_path = os.path.join(OUTPUT_IMAGE_16BIT_DIR, image_16bit_filename)
            shutil.copy2(image_16bit_path, output_16bit_path)
            print(f"[{i}] Disalin 16bit: {image_16bit_filename}")
        else:
            print(f"[{i}] ❌ Citra 16bit '{image_16bit_filename}' tidak ditemukan.")

        # 3. Salin citra 8bit (.tif)
        image_8bit_filename = f"{base_name}.tif"
        image_8bit_path = os.path.join(INPUT_IMAGE_8BIT_DIR, image_8bit_filename)
        if os.path.exists(image_8bit_path):
            output_8bit_path = os.path.join(OUTPUT_IMAGE_8BIT_DIR, image_8bit_filename)
            shutil.copy2(image_8bit_path, output_8bit_path)
            print(f"[{i}] Disalin 8bit: {image_8bit_filename}")
        else:
            print(f"[{i}] ❌ Citra 8bit '{image_8bit_filename}' tidak ditemukan.")

    print(f"\nSelesai! {len(selected_labels)} label dan citra terkait telah disalin.")

if __name__ == "__main__":
    main()
