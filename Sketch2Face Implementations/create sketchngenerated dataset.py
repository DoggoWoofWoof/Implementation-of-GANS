import os
import glob
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
import glob
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Load Model
g_model = load_model("C:/Users/Swastik/Desktop/generator_epoch_13.h5")

# Function to process a single image
def process_image(sketch_path):
    # Load and resize the image
    sketch = load_img(sketch_path, target_size=(128, 128))

    # Convert to numpy array
    sketch = img_to_array(sketch)
    norm_sketch = (sketch.copy() - 127.5) / 127.5
    g_img = g_model.predict(np.expand_dims(norm_sketch, 0))[0]
    g_img = g_img * 127.5 + 127.5

    sketch = cv2.resize(sketch, (200, 250))
    g_img = cv2.resize(g_img, (200, 250))

    # Create a mask for the sketch border
    sketch_gray = cv2.cvtColor(sketch.astype('uint8'), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(sketch_gray, 100, 200)

    # Slightly dilate the edges to make them a bit thicker
    kernel = np.ones((2, 2), np.uint8)  # Reduced kernel size
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Create the final image by overlaying the generated image on the sketch
    final_img = sketch.copy()
    mask = edges == 0
    final_img[mask] = g_img[mask]

    return final_img

def create_sketchngenerated_dataset(input_dir, output_dir, glob_pattern):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the file names
    filenames = sorted(glob.glob(glob_pattern))

    for filename in filenames:
        # Process the image
        processed_img = process_image(filename)

        # Save the processed image
        base_name = os.path.basename(filename)
        output_path = os.path.join(output_dir, base_name)
        cv2.imwrite(output_path, cv2.cvtColor(processed_img.astype('uint8'), cv2.COLOR_RGB2BGR))


# First, create the sketchngenerated dataset
create_sketchngenerated_dataset(
    input_dir='C:/Users/Swastik/Desktop/Twin Model Dataset (CUHK 13 epoch)/Augmented sketch',
    output_dir='C:/Users/Swastik/Desktop/Twin Model Dataset (CUHK 13 epoch)/Augmented sketchngenerated',
    glob_pattern='C:/Users/Swastik/Desktop/Twin Model Dataset (CUHK 13 epoch)/Augmented sketch/*.jpg'
)
