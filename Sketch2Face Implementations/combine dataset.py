import torch
import torchvision.transforms as T
from PIL import Image
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from scipy.spatial.distance import cosine
import os
import os
import shutil
from PIL import Image
import imagehash
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging

# Set up logging with a simpler format
logging.basicConfig(level=logging.INFO, format='%(message)s')

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

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

def deep_lab_segmentation(image):
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    return output_predictions.byte().cpu().numpy()

def refine_mask(mask, kernel_size=(9, 9)):
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def create_soft_mask(mask, kernel_size=(9, 9)):
    mask = cv2.GaussianBlur(mask.astype(np.float32), kernel_size, sigmaX=0, sigmaY=0)
    mask = np.clip(mask, 0, 1)
    return mask

def apply_grabcut(image_cv, mask):
    grabcut_mask = np.where(mask == 1, 3, 2).astype(np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (1, 1, image_cv.shape[1] - 2, image_cv.shape[0] - 2)  # slightly smaller rect to avoid boundary issues
    cv2.grabCut(image_cv, grabcut_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    refined_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype(np.uint8)

    return refined_mask

def remove_background_dl(image_cv):
    image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    mask = deep_lab_segmentation(image)
    mask = np.where(mask == 15, 1, 0).astype(np.uint8)  # 15 is the label for 'person'

    foreground_ratio = np.sum(mask) / mask.size

    if foreground_ratio < 0.3:
        logging.warning(f"High foreground removal detected: {foreground_ratio * 100:.2f}% of image removed. Refining mask.")
        mask = refine_mask(mask, kernel_size=(3, 3))
        mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)
    else:
        mask = refine_mask(mask, kernel_size=(9, 9))
    
    foreground_ratio_after_refinement = np.sum(mask) / mask.size
    if foreground_ratio_after_refinement < 0.1:
        logging.warning(f"Second refinement removed too much foreground: {foreground_ratio_after_refinement * 100:.2f}% of image removed. Softening further.")
        mask = refine_mask(mask, kernel_size=(2, 2))

    if np.sum(mask) / mask.size < 0.2:
        logging.warning("Applying GrabCut due to insufficient foreground detection.")
        mask = apply_grabcut(image_cv, mask)

    soft_mask = create_soft_mask(mask, kernel_size=(9, 9))
    soft_mask = cv2.resize(soft_mask, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Convert original image to BGRA (BGR + Alpha)
    result = cv2.cvtColor(image_cv, cv2.COLOR_BGR2BGRA)
    
    # Apply the soft mask to the alpha channel
    result[:, :, 3] = (soft_mask * 255).astype(np.uint8)

    # Convert result back to RGB from BGRA
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
    
    return result_rgb

def convert_image(input_path, output_path, grayscale=False, background=False):
    try:
        logging.info(f"Starting conversion for {input_path}")

        # Load the image
        img = Image.open(input_path)

        # Convert to RGBA if not already in that mode
        if img.mode not in ['RGB', 'RGBA', 'L']:
            img = img.convert('RGBA')

        # Convert PIL image to OpenCV format (numpy array)
        image_cv = np.array(img)

        # Perform background removal if requested
        if background:
            image_cv = cv2.imread(input_path)
            result = remove_background_dl(image_cv)  # Remove background
            img = Image.fromarray(result, 'RGBA')  # Convert back to PIL format with alpha channel
            logging.info("Removed background.")

        # Convert to grayscale if requested
        if grayscale:
            if img.mode == 'RGBA':
                # Convert RGBA to grayscale and keep alpha channel
                img = img.convert("LA")
            elif img.mode == 'RGB':
                # Convert RGB to grayscale
                img = img.convert("L")
            logging.info("Image converted to grayscale.")

        # Save the image
        img.save(output_path, 'PNG')
        logging.info(f"Image saved to {output_path}")

        return True

    except Exception as e:
        logging.error(f"Error converting image {input_path}: {str(e)}")
        return False

def process_and_copy_files(args):
    photo_path, sketch_path, new_photo_path, new_sketch_path, grayscale, background = args
    if convert_image(photo_path, new_photo_path, grayscale, background) and convert_image(sketch_path, new_sketch_path):
        logging.info(f"Processed: {photo_path} -> {new_photo_path}, {sketch_path} -> {new_sketch_path}")
        return True
    return False

def find_matching_file(base_path, extensions):
    for ext in extensions:
        full_path = base_path + ext
        if os.path.exists(full_path):
            return full_path
    return None

def get_user_defined_pattern(photo_subdir, sketch_subdir, last_patterns=None):
    use_same_pattern = False

    if last_patterns:
        use_same_pattern_input = input(f"Use last pattern '{last_patterns[0]}' for photos and '{last_patterns[1]}' for sketches? (y/n): ").strip().lower()
        if use_same_pattern_input == 'y':
            use_same_pattern = True

    if use_same_pattern and last_patterns:
        photo_pattern, sketch_pattern = last_patterns
    else:
        logging.info(f"Enter pattern for directories:\nPhoto: {photo_subdir}\nSketch: {sketch_subdir}")
        photo_pattern = input("photos pattern (e.g., photos:m2-038-01.jpg): ").strip()
        sketch_pattern = input("sketches pattern (e.g., sketches:M2-038-01-sz1.jpg): ").strip()

    return photo_pattern, sketch_pattern

def rename_files_sequentially(photo_dir, sketch_dir, supported_extensions):
    photo_files = sorted([f for f in os.listdir(photo_dir) if any(f.lower().endswith(ext) for ext in supported_extensions)])
    sketch_files = sorted([f for f in os.listdir(sketch_dir) if any(f.lower().endswith(ext) for ext in supported_extensions)])
    
    if len(photo_files) != len(sketch_files):
        logging.error("Number of photos and sketches files do not match. Cannot proceed with sequential renaming.")
        return False

    for i, (photo_file, sketch_file) in enumerate(zip(photo_files, sketch_files), 1):
        new_photo_name = f"{i}.jpg"
        new_sketch_name = f"{i}.jpg"
        
        os.rename(os.path.join(photo_dir, photo_file), os.path.join(photo_dir, new_photo_name))
        os.rename(os.path.join(sketch_dir, sketch_file), os.path.join(sketch_dir, new_sketch_name))
    
    logging.info("Files renamed sequentially.")
    return True

def count_files(directory, extensions):
    total_files = 0
    for root, _, files in os.walk(directory):
        file_count = sum(1 for f in files if any(f.lower().endswith(ext) for ext in extensions))
        logging.info(f"Directory: {root}, Files: {file_count}")
        total_files += file_count
    return total_files

def find_matching_sketch(photo_base_name, sketch_dir, supported_extensions):
    for ext in supported_extensions:
        sketch_path = os.path.join(sketch_dir, f"{photo_base_name}{ext}")
        if os.path.exists(sketch_path):
            return sketch_path
    return None

# Initialize MTCNN and Resnet for face detection and embeddings
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(mtcnn.device)

def get_face_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    faces = mtcnn(image)
    if faces is None:
        return None
    embeddings = resnet(faces)
    return embeddings[0].detach().cpu().numpy() if embeddings is not None else None

def calculate_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    return 1 - cosine(embedding1, embedding2)

def process_photo(args):
    photo_path, sketch_path, combined_dir, non_colored_dir, seen_embeddings, seen_files, seen_hashes, grayscale, duplicate_mapping = args

    try:
        with Image.open(photo_path) as img:
            current_hash = imagehash.average_hash(img, hash_size=8)
        is_duplicate = False
        original_photo = None

        # Check for duplicates based on image hash
        for seen_hash, seen_file in zip(seen_hashes, seen_files):
            if isinstance(seen_hash, imagehash.ImageHash) and (current_hash - seen_hash) < 5:
                if images_are_identical(photo_path, seen_file):
                    is_duplicate = True
                    original_photo = seen_file
                    break

        if not is_duplicate:
            current_embedding = get_face_embedding(photo_path)
            if current_embedding is None:
                logging.info(f"No faces found in {photo_path}. Skipping face-based duplicate check.")
                return 'copy', (None, photo_path, sketch_path)

            # Check for duplicates based on facial embeddings
            for seen_embedding, seen_file in zip(seen_embeddings, seen_files):
                if calculate_similarity(current_embedding, seen_embedding) >= 0.85:  # If faces are similar
                    is_duplicate = True
                    original_photo = seen_file
                    break

        if is_duplicate:
            logging.info(f"Duplicate found: {photo_path} is a duplicate of {original_photo}")
            os.remove(photo_path)
            if os.path.exists(sketch_path):
                os.remove(sketch_path)
            duplicate_mapping.setdefault(original_photo, []).append(photo_path)
            return 'duplicate', None            

        if not grayscale and is_grayscale(photo_path):
            logging.info(f"Grayscale: {photo_path}")
            new_photo_path = os.path.join(non_colored_dir, 'photos', os.path.basename(photo_path))
            new_sketch_path = os.path.join(non_colored_dir, 'sketches', os.path.basename(sketch_path))
            os.makedirs(os.path.dirname(new_photo_path), exist_ok=True)
            os.makedirs(os.path.dirname(new_sketch_path), exist_ok=True)
            shutil.move(photo_path, new_photo_path)
            if os.path.exists(sketch_path):
                shutil.move(sketch_path, new_sketch_path)
            return 'grayscale', None

        logging.info(f"Processing: {photo_path} with sketches {sketch_path}")
        return 'copy', (current_embedding, photo_path, sketch_path, current_hash)

    except Exception as e:
        logging.error(f"Error processing {photo_path}: {str(e)}")
        return 'error', None

def process_directory(base_dir, combined_dir):
    os.makedirs(os.path.join(combined_dir, 'photos'), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, 'sketches'), exist_ok=True)

    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    background_option = input("Do you want to remove the background of images? (y/n): ").strip().lower()
    background = background_option == 'y'

    grayscale_option = input("Do you want to convert images to grayscale? (y/n): ").strip().lower()
    grayscale = grayscale_option == 'y'

    if not grayscale:
        non_colored_dir = 'C:/Users/Swastik/Desktop/non_colored'
        os.makedirs(os.path.join(non_colored_dir, 'photos'), exist_ok=True)
        os.makedirs(os.path.join(non_colored_dir, 'sketches'), exist_ok=True)
    else:
        non_colored_dir = None

    all_files = []
    unmatched_files = []

    # First pass: Find photos-sketches pairs based on file names
    for root, dirs, files in os.walk(base_dir):
        logging.info(f"Checking directory: {root}")

        if 'photos' in root.lower():
            photo_dir = root
            sketch_dir = root.replace('photos', 'sketches')

            if os.path.exists(sketch_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        photo_path = os.path.join(photo_dir, file)
                        photo_base_name = os.path.splitext(file)[0]
                        sketch_path = find_matching_sketch(photo_base_name, sketch_dir, supported_extensions)
                        
                        if sketch_path:
                            all_files.append((photo_path, sketch_path))
                        else:
                            unmatched_files.append(photo_path)
                            logging.info(f"No sketches found for: {photo_path}")
            else:
                logging.warning(f"Missing sketches directory: {sketch_dir}")

    logging.info(f"Unmatched photos: {len(unmatched_files)}")
    
    # If no pairs, proceed with pattern matching or sequential renaming
    if not all_files:
        logging.info("No pairs found. Offering additional options...")

        choice = input("Would you like to enter a pattern or rename files sequentially?\n1. Enter pattern\n2. Rename sequentially\nChoose (1/2): ").strip()

        unmatched_dirs = set()
        for photo_path in unmatched_files:
            photo_dir = os.path.dirname(photo_path)
            sketch_dir = photo_dir.replace('photos', 'sketches')
            
            if os.path.exists(photo_dir) and os.path.exists(sketch_dir):
                unmatched_dirs.add((photo_dir, sketch_dir))

        if unmatched_dirs:
            logging.info(f"Unmatched directories: {len(unmatched_dirs)}")

        if choice == '1':
            last_patterns = None
            
            for photo_dir, sketch_dir in unmatched_dirs:
                photo_pattern, sketch_pattern = get_user_defined_pattern(photo_dir, sketch_dir, last_patterns)
                last_patterns = (photo_pattern, sketch_pattern)
                
                for file in os.listdir(photo_dir):
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        photo_path = os.path.join(photo_dir, file)
                        sketch_base = os.path.join(sketch_dir, sketch_pattern.replace(photo_pattern, os.path.splitext(file)[0]))
                        sketch_path = find_matching_file(sketch_base, supported_extensions)
                        
                        if sketch_path:
                            all_files.append((photo_path, sketch_path))
                        else:
                            logging.warning(f"No sketches found for: {photo_path}")

        elif choice == '2':
            for photo_dir, sketch_dir in unmatched_dirs:
                if not rename_files_sequentially(photo_dir, sketch_dir, supported_extensions):
                    logging.error("Error in renaming files sequentially. Aborting process.")
                    return

                for file in os.listdir(photo_dir):
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        photo_path = os.path.join(photo_dir, file)
                        sketch_path = os.path.join(sketch_dir, file)
                        if os.path.exists(sketch_path):
                            all_files.append((photo_path, sketch_path))

    logging.info(f"photos-sketches pairs found: {len(all_files)}")

    seen_embeddings = []
    seen_files = []
    seen_hashes = []
    files_to_copy = []
    duplicate_mapping = {}
    grayscale_files=[]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        for photo_path, sketch_path in all_files:
            result, data = process_photo((photo_path, sketch_path, combined_dir, non_colored_dir, seen_embeddings, seen_files, seen_hashes, grayscale, duplicate_mapping))
            if result == 'copy':
                files_to_copy.append((data[1], data[2]))
                if data[0] is not None:  # Only add non-None embeddings
                    seen_embeddings.append(data[0])
                    seen_files.append(data[1])
                    seen_hashes.append(data[3])

    logging.info(f"Files to copy: {len(files_to_copy)}")

    # Copy files
    successful_copies = 0
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        copy_args = [
            (
                photo_path,
                sketch_path,
                os.path.join(combined_dir, 'photos', f"{i+1}.png"),
                os.path.join(combined_dir, 'sketches', f"{i+1}.png"),
                grayscale,
                background
            )
            for i, (photo_path, sketch_path) in enumerate(files_to_copy)
        ]

        for success in executor.map(process_and_copy_files, copy_args):
            if success:
                successful_copies += 1

    logging.info(f"Processing completed. Successful copies: {successful_copies}")
    logging.info("Duplicate mapping:")
    for original, duplicates in duplicate_mapping.items():
        logging.info(f"{original} has duplicates: {', '.join(duplicates)}")
        
    logging.info(f"Total Files {len(all_files)}")
    logging.info(f"Duplicates {len(duplicate_mapping)}")
    logging.info(f"Grayscale {len(grayscale_files)}")
    logging.info(f"Files Copied {successful_copies}")

def normalize_path(path):
    # Normalize path to use the correct separator for the current operating system
    return os.path.normpath(path)

def process_path_input(path):
    # Strip leading and trailing quotes
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    return path

if __name__ == '__main__':
    # Take user input for directories
    base_dir_input = input('Enter the base directory path (enclosed in double quotes): ')
    combined_dir_input = input('Enter the combined directory path (enclosed in double quotes): ')

    # Process and normalize the paths
    base_dir = process_path_input(base_dir_input)
    combined_dir = process_path_input(combined_dir_input)

    base_dir = normalize_path(base_dir)
    combined_dir = normalize_path(combined_dir)
    process_directory(base_dir, combined_dir)