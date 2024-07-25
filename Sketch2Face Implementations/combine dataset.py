import os
import shutil
from PIL import Image
import imagehash
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_grayscale(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    is_gray = np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 1] == img[:, :, 2])
    return is_gray

def images_are_identical(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return False
    return np.all(img1 == img2)

def convert_image(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            img.save(output_path, 'PNG')
        return True
    except Exception as e:
        logging.error(f"Error converting image {input_path}: {str(e)}")
        return False

def process_and_copy_files(args):
    photo_path, sketch_path, new_photo_path, new_sketch_path = args
    if convert_image(photo_path, new_photo_path) and convert_image(sketch_path, new_sketch_path):
        logging.info(f"Processed and copied: {photo_path} -> {new_photo_path}, {sketch_path} -> {new_sketch_path}")
        return True
    return False

def find_matching_file(base_path, extensions):
    for ext in extensions:
        full_path = base_path + ext
        if os.path.exists(full_path):
            return full_path
    return None

def process_photo(args):
    photo_path, sketch_path, combined_dir, non_colored_dir, seen_hashes, seen_files = args

    try:
        with Image.open(photo_path) as img:
            current_hash = imagehash.average_hash(img, hash_size=8)
        
        is_duplicate = False
        for seen_hash, seen_file in zip(seen_hashes, seen_files):
            if current_hash - seen_hash < 5:  # If hashes are similar, do pixel-by-pixel comparison
                if images_are_identical(photo_path, seen_file):
                    is_duplicate = True
                    break

        if is_duplicate:
            logging.info(f"Duplicate found: {photo_path}")
            os.remove(photo_path)
            if os.path.exists(sketch_path):
                os.remove(sketch_path)
            return 'duplicate', None

        if is_grayscale(photo_path):
            logging.info(f"Grayscale image found: {photo_path}")
            new_photo_path = os.path.join(non_colored_dir, 'photos', os.path.basename(photo_path))
            new_sketch_path = os.path.join(non_colored_dir, 'sketches', os.path.basename(sketch_path))
            os.makedirs(os.path.dirname(new_photo_path), exist_ok=True)
            os.makedirs(os.path.dirname(new_sketch_path), exist_ok=True)
            shutil.move(photo_path, new_photo_path)
            if os.path.exists(sketch_path):
                shutil.move(sketch_path, new_sketch_path)
            return 'grayscale', None

        logging.info(f"Processing: {photo_path} with sketch {sketch_path}")
        return 'copy', (current_hash, photo_path, sketch_path)

    except Exception as e:
        logging.error(f"Error processing {photo_path}: {str(e)}")
        return 'error', None

def process_directory(base_dir, combined_dir, non_colored_dir):
    os.makedirs(os.path.join(combined_dir, 'photos'), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, 'sketches'), exist_ok=True)
    os.makedirs(os.path.join(non_colored_dir, 'photos'), exist_ok=True)
    os.makedirs(os.path.join(non_colored_dir, 'sketches'), exist_ok=True)

    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    all_files = []
    for root, _, files in os.walk(base_dir):
        if 'photos' in root:
            photo_dir = root
            sketch_dir = root.replace('photos', 'sketches')
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    photo_path = os.path.join(photo_dir, file)
                    sketch_base = os.path.splitext(os.path.join(sketch_dir, file))[0]
                    sketch_path = find_matching_file(sketch_base, supported_extensions)
                    if sketch_path:
                        all_files.append((photo_path, sketch_path))
                    else:
                        logging.info(f"No corresponding sketch found for: {photo_path}")

    logging.info(f"Found {len(all_files)} photo-sketch pairs to process")

    seen_hashes = []
    seen_files = []
    files_to_copy = []

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for photo_path, sketch_path in all_files:
            result, data = process_photo((photo_path, sketch_path, combined_dir, non_colored_dir, seen_hashes, seen_files))
            if result == 'copy':
                files_to_copy.append((data[1], data[2]))
                seen_hashes.append(data[0])
                seen_files.append(data[1])

    logging.info(f"Files to copy: {len(files_to_copy)}")

    # Copy files
    successful_copies = 0
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        copy_args = [
            (
                photo_path,
                sketch_path,
                os.path.join(combined_dir, 'photos', f"{i+1}.png"),
                os.path.join(combined_dir, 'sketches', f"{i+1}.png")
            )
            for i, (photo_path, sketch_path) in enumerate(files_to_copy)
        ]
        results = executor.map(process_and_copy_files, copy_args)
        successful_copies = sum(results)

    return successful_copies

def main():
    base_dir = "C:/Users/Swastik/Desktop/dataset/dataset"
    combined_dir = "C:/Users/Swastik/Desktop/combined"
    non_colored_dir = "C:/Users/Swastik/Desktop/non_colored"

    logging.info("Starting processing...")
    logging.info(f"Base directory: {base_dir}")
    logging.info(f"Combined directory: {combined_dir}")
    logging.info(f"Non-colored directory: {non_colored_dir}")

    if not os.path.exists(base_dir):
        logging.error(f"Base directory does not exist: {base_dir}")
        return

    total_processed = process_directory(base_dir, combined_dir, non_colored_dir)

    logging.info(f"All processing completed. Total files successfully processed: {total_processed}")

if __name__ == "__main__":
    main()