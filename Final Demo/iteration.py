import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import structural_similarity as ssim
import imageio.v2 as imageio

# Function to load and preprocess image
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    norm_img = (img - 127.5) / 127.5
    return img, norm_img

# Function to generate image using the model
def generate_image(model, norm_img):
    g_img = model.predict(np.expand_dims(norm_img, 0))[0]
    g_img = g_img * 127.5 + 127.5
    return np.clip(g_img, 0, 255).astype('uint8')

# Function to load image
def load_image(image_path):
    return cv2.imread(image_path)

# Function to compute L2 norm
def compute_l2(img1, img2):
    return np.mean(np.square(img1 - img2))

# Function to compute SSIM
def compute_ssim(img1, img2):
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

# Function to resize images to a target shape
def resize_image(image, target_size=(128, 128)):
    return cv2.resize(image, target_size)

# Function to display and save images
def display_images(img, g_img, target, preprocessed_img, iteration=None, best_iteration=None, best_metrics=None):
    img_resized = resize_image(img, (200, 250))
    g_img_resized = resize_image(g_img, (200, 250))
    target_resized = resize_image(target, (200, 250))
    preprocessed_img_resized = resize_image((preprocessed_img * 127.5 + 127.5).astype('uint8'), (200, 250))

    f = plt.figure(num=None, figsize=(16, 6), dpi=80)
    ax1 = f.add_subplot(1, 4, 1)
    plt.imshow(img_resized.astype('uint8'))
    ax2 = f.add_subplot(1, 4, 2)
    plt.imshow(preprocessed_img_resized)
    ax3 = f.add_subplot(1, 4, 3)
    plt.imshow(g_img_resized.astype('uint8'))
    ax4 = f.add_subplot(1, 4, 4)
    plt.imshow(target_resized.astype('uint8'))
    ax1.set_title('Original Sketch')
    ax2.set_title('Preprocessed Image')
    ax3.set_title('Generated Image')
    ax4.set_title('Target Image')

    if iteration is not None:
        plt.suptitle(f'Iteration: {iteration}', fontsize=16)
    elif best_iteration is not None:
        plt.suptitle(f'Best Result at Iteration: {best_iteration + 1}', fontsize=16)

    if best_metrics is not None:
        plt.figtext(0.5, 0.01, f'Best L2-norm: {best_metrics[0]:.4f} :: Best SSIM: {best_metrics[1]:.4f}', ha='center', fontsize=12)

    # Save the figure as an image
    filename = f'iteration_{iteration}.png' if iteration is not None else 'best_image.png'
    plt.savefig(filename)
    plt.close(f)
    return filename

def process_image(sketch, g_img):
    sketch = resize_image(sketch, (128, 128))
    g_img = resize_image(g_img, (128, 128))

    # Create a mask for the sketch border
    sketch_gray = cv2.cvtColor(sketch.astype('uint8'), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(sketch_gray, 100, 200)

    # Slightly dilate the edges to make them a bit thicker
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Create the final image by overlaying the generated image on the sketch
    final_img = sketch.copy()
    mask = edges == 0
    final_img[mask] = g_img[mask]

    return final_img

#  Main execution with iterative processing and model switching
def main(sketch_path, target_path, model_path1, model_path2, max_iterations=5, no_improvement_limit=5):
    # Load models
    g_model1 = load_model(model_path1)
    g_model2 = load_model(model_path2)

    # Load and preprocess original sketch image
    original_sketch, norm_original_sketch = load_and_preprocess_image(sketch_path)

    # Generate initial image using g_model1
    initial_generated = generate_image(g_model1, norm_original_sketch)

    # Load target image
    target = cv2.cvtColor(load_image(target_path), cv2.COLOR_BGR2RGB)
    target_resized = resize_image(target, (256, 256))

    # Compute initial metrics
    best_l2 = compute_l2(resize_image(initial_generated, (256, 256)), target_resized)
    best_ssim = compute_ssim(resize_image(initial_generated, (256, 256)), target_resized)
    best_image = initial_generated.copy()
    no_improvement_count = 0
    best_iteration = 0

    print(f"Initial L2-norm: {best_l2:.4f} :: SSIM: {best_ssim:.4f}")

    # Display initial images
    display_images(original_sketch, initial_generated, target_resized, norm_original_sketch, iteration=0)

    # Iteratively process the image using g_model2
    current_input = initial_generated
    filenames = []  # Initialize the list to store filenames
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}...")

        # Apply process_image to the current input
        processed_img = process_image(original_sketch, current_input)

        # Generate new image using g_model2
        norm_processed_img = (processed_img - 127.5) / 127.5
        generated_img = generate_image(g_model2, norm_processed_img)

        # Compute metrics for the generated image
        current_l2 = compute_l2(resize_image(generated_img, (256, 256)), target_resized)
        current_ssim = compute_ssim(resize_image(generated_img, (256, 256)), target_resized)

        print(f"Iteration {iteration + 1} L2-norm: {current_l2:.4f} :: SSIM: {current_ssim:.4f}")

        if current_l2 < best_l2 and current_ssim > best_ssim:
            best_l2 = current_l2
            best_ssim = current_ssim
            best_image = generated_img.copy()
            best_iteration = iteration
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Display images after processing
        filename = display_images(original_sketch, generated_img, target_resized, norm_processed_img, iteration=iteration+1)
        filenames.append(filename)  # Append the filename to the list

        # Save the processed image of this iteration
        cv2.imwrite(f'processed_image_{iteration}.png', generated_img)

        # Update current input for next iteration
        current_input = generated_img

        if no_improvement_count >= no_improvement_limit:
            print(f"No improvement for {no_improvement_limit} iterations. Stopping.")
            break

    # Display and save the best image
    best_filename = display_images(original_sketch, best_image, target_resized, norm_original_sketch,
                   best_iteration=best_iteration, best_metrics=(best_l2, best_ssim))
    filenames.append(best_filename)

    # Create the GIF with each frame lasting 5000 milliseconds (5 seconds)
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimwrite('iterations.gif', images, duration=1000)  # Duration in milliseconds

    print(f"Processing complete. Best image saved as 'best_image.png'. Best result at iteration {best_iteration + 1}. GIF saved as 'iterations.gif'.")

# Example usage
sketch_path = "C:/Users/Swastik/Desktop/Final Demo/bsketch.jpg"
target_path = "C:/Users/Swastik/Desktop/Final Demo/bphoto.jpg"
model_path1 = "C:/Users/Swastik/Desktop/Final Demo/model/generator_epoch_27.h5"
model_path2 = "C:/Users/Swastik/Desktop/generator_epoch_35.h5"
main(sketch_path, target_path, model_path1, model_path2)
