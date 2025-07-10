import os
import numpy as np
import rasterio
from tqdm import tqdm
from matplotlib import pyplot as plt

# Konfigurasi
PATCH_8BIT_DIR = 'patch_images_8bit'
PATCH_16BIT_DIR = 'patch_images_16bit'
LABEL_INDEX_DIR = 'patch_labels_index'

WHITE_THRESHOLD = 250
BORDER_WIDTH = 5
STRICT_MODE = True
REMOVE = True

DELETED_LOG_FILE = 'deleted_files.txt'

def has_complete_white_border(patch_path):
    """Cek apakah patch memiliki border putih"""
    try:
        with rasterio.open(patch_path) as src:
            data = src.read()
            combined = np.all(data > WHITE_THRESHOLD, axis=0)

            def check_full_border(region): return np.all(region)
            def check_partial_border(region): return np.any(region)

            check_func = check_full_border if STRICT_MODE else check_partial_border

            borders = [
                combined[:BORDER_WIDTH, :],
                combined[-BORDER_WIDTH:, :],
                combined[:, :BORDER_WIDTH],
                combined[:, -BORDER_WIDTH:]
            ]

            return any(check_func(border) for border in borders)
    except Exception as e:
        print(f"Error processing {patch_path}: {str(e)}")
        return False

def delete_related_files(base_filename, log_file):
    """Hapus file 16bit dan label index yang sesuai nama dasarnya"""
    # Hapus citra 16 bit
    path_16bit = os.path.join(PATCH_16BIT_DIR, base_filename)
    if os.path.exists(path_16bit):
        os.remove(path_16bit)
        log_file.write(f"[deleted] 16bit: {path_16bit}\n")

    # Hapus label index
    label_filename = base_filename.replace('.tif', '_5label.png')
    label_path = os.path.join(LABEL_INDEX_DIR, label_filename)
    if os.path.exists(label_path):
        os.remove(label_path)
        log_file.write(f"[deleted] label index: {label_path}\n")

def main():
    total_patches = 0
    deleted = 0

    if REMOVE:
        log_file = open(DELETED_LOG_FILE, 'w')

    print("Memulai pengecekan patch dengan border putih...")

    patch_files = [f for f in os.listdir(PATCH_8BIT_DIR) if f.endswith('.tif')]

    for patch_file in tqdm(patch_files, desc="Memeriksa patch"):
        patch_path = os.path.join(PATCH_8BIT_DIR, patch_file)
        total_patches += 1

        if has_complete_white_border(patch_path):
            if REMOVE:
                # Preview (opsional)
                with rasterio.open(patch_path) as src:
                    rgb = np.stack([src.read(3), src.read(2), src.read(1)], axis=-1)
                    plt.imshow(rgb)
                    plt.title(f"Border putih: {patch_file}")
                    plt.show()

                # Hapus 8bit
                os.remove(patch_path)
                log_file.write(f"[deleted] 8bit: {patch_path}\n")

                # Hapus file terkait
                delete_related_files(patch_file, log_file)

            deleted += 1

    if REMOVE:
        log_file.close()

    print(f"\nSelesai. {deleted} file dihapus dari {total_patches} total patch.")
    print(f"Detail tersimpan di '{DELETED_LOG_FILE}'")

if __name__ == "__main__":
    if not os.path.exists(PATCH_8BIT_DIR):
        print(f"Folder {PATCH_8BIT_DIR} tidak ditemukan!")
        exit()
    main()
