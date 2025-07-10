import os
import numpy as np
from PIL import Image
import math

# Parameter yang bisa disesuaikan
INPUT_DIR = 'test/selected_label_index'  # Folder berisi 15 gambar terpilih
OUTPUT_DIR = 'test/patch_label_index'   # Folder untuk menyimpan patch
PATCH_SIZE = 256                 # Ukuran patch yang diinginkan

def create_patches(image_path, output_folder):
    """Membuat patch 256x256 dari sebuah gambar"""
    try:
        # Buka gambar
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Dapatkan dimensi gambar
        height, width = img_array.shape[0], img_array.shape[1]
        
        # Hitung jumlah patch vertikal dan horizontal
        num_patches_h = math.ceil(width / PATCH_SIZE)
        num_patches_v = math.ceil(height / PATCH_SIZE)
        
        # Buat patch
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        patch_count = 0
        
        for i in range(num_patches_v):
            for j in range(num_patches_h):
                # Tentukan koordinat patch
                y_start = i * PATCH_SIZE
                y_end = min((i + 1) * PATCH_SIZE, height)
                x_start = j * PATCH_SIZE
                x_end = min((j + 1) * PATCH_SIZE, width)
                
                # Ekstrak patch
                patch = img_array[y_start:y_end, x_start:x_end]
                
                # Jika patch lebih kecil dari 256x256, kita bisa:
                # 1. Discard patch ini, atau
                # 2. Pad dengan nilai tertentu (di sini kita memilih opsi 1)
                if patch.shape[0] == PATCH_SIZE and patch.shape[1] == PATCH_SIZE:
                    # Simpan patch
                    patch_img = Image.fromarray(patch)
                    patch_filename = f"{base_filename}_patch_{i}_{j}.png"
                    patch_img.save(os.path.join(output_folder, patch_filename))
                    patch_count += 1
        
        return patch_count
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return 0

def main():
    # Buat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_patches = 0
    
    print("Memulai proses pembuatan patch 256x256...")
    
    # Proses setiap gambar di folder input
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(INPUT_DIR, filename)
            print(f"Memproses gambar: {filename}")
            
            # Buat patch untuk gambar ini
            num_patches = create_patches(image_path, OUTPUT_DIR)
            total_patches += num_patches
            print(f"  -> Berhasil membuat {num_patches} patch")
    
    print(f"\nSelesai! Total {total_patches} patch berukuran {PATCH_SIZE}x{PATCH_SIZE} telah dibuat di folder '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()