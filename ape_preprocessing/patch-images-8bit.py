import os
import numpy as np
from PIL import Image
import math
import rasterio

# Konfigurasi
INPUT_DIR = 'selected_images_gid_8bit_10'  # Folder gambar NIR-RGB
OUTPUT_DIR = 'patch_images_gid_8bit_10'  # Folder output
PATCH_SIZE = 256  # Ukuran patch
BANDS = ['NIR', 'Red', 'Green', 'Blue']  # Urutan band

def create_nir_rgb_patches(image_path, output_folder):
    """Membuat patch 256x256 dari citra NIR-RGB"""
    try:
        # Buka citra dengan rasterio untuk mempertahankan multi-band
        with rasterio.open(image_path) as src:
            # Baca semua band
            img_data = src.read()  # Shape: (bands, height, width)
            
            # Pastikan citra memiliki 4 band
            if img_data.shape[0] != 4:
                print(f"Citra {os.path.basename(image_path)} tidak memiliki 4 band")
                return 0
            
            height, width = img_data.shape[1], img_data.shape[2]
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            patch_count = 0
            
            # Hitung jumlah patch
            num_patches_h = math.ceil(width / PATCH_SIZE)
            num_patches_v = math.ceil(height / PATCH_SIZE)
            print(f"Memproses {base_filename}: {num_patches_v} patch vertikal, {num_patches_h} patch horizontal")
            
            for i in range(num_patches_v):
                for j in range(num_patches_h):
                    # Tentukan koordinat patch
                    y_start = i * PATCH_SIZE
                    y_end = min((i + 1) * PATCH_SIZE, height)
                    x_start = j * PATCH_SIZE
                    x_end = min((j + 1) * PATCH_SIZE, width)
                    
                    # Ekstrak patch dari semua band
                    patch = img_data[:, y_start:y_end, x_start:x_end]
                    
                    # Hanya simpan patch yang berukuran penuh
                    if patch.shape[1] == PATCH_SIZE and patch.shape[2] == PATCH_SIZE:
                        # Simpan patch sebagai file .tif multi-band
                        patch_filename = f"{base_filename}_patch_{i}_{j}.tif"
                        patch_path = os.path.join(output_folder, patch_filename)
                        
                        # Simpan dengan metadata yang sesuai
                        with rasterio.open(
                            patch_path,
                            'w',
                            driver='GTiff',
                            height=PATCH_SIZE,
                            width=PATCH_SIZE,
                            count=4,
                            dtype=img_data.dtype,
                            crs=src.crs,
                            transform=rasterio.windows.transform(
                                rasterio.windows.Window(x_start, y_start, PATCH_SIZE, PATCH_SIZE),
                                src.transform
                            )
                        ) as dst:
                            dst.write(patch)
                        
                        patch_count += 1
            
            return patch_count
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return 0

def main():
    # Buat folder output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_patches = 0
    
    print("Memulai proses pembuatan patch NIR-RGB 256x256...")
    
    # Proses setiap gambar
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith('.tif'):
            image_path = os.path.join(INPUT_DIR, filename)
            print(f"Memproses: {filename}")
            
            num_patches = create_nir_rgb_patches(image_path, OUTPUT_DIR)
            total_patches += num_patches
            print(f"  -> Berhasil membuat {num_patches} patch")
    
    print(f"\nSelesai! Total {total_patches} patch NIR-RGB berukuran {PATCH_SIZE}x{PATCH_SIZE} dibuat di '{OUTPUT_DIR}'")

if __name__ == "__main__":
    # Pastikan library yang diperlukan terinstall
    try:
        import rasterio
    except ImportError:
        print("Silakan install rasterio terlebih dahulu:")
        print("pip install rasterio")
        exit()
    
    main()