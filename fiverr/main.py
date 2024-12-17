import cv2
import numpy as np
import os
from tqdm import tqdm


def composite_image(image_path, mask_path, output_path, alpha=0.5):
    # Load the actual image and mask
    actual_image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if actual_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if mask_image is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Create the masked image
    masked_image = cv2.bitwise_and(actual_image, actual_image, mask=mask_image)

    # Create the overlay image
    overlay_image = cv2.addWeighted(masked_image, alpha, actual_image, 1 - alpha, 0)

    # Create image where mask is removed
    mask_removed = cv2.bitwise_and(actual_image, actual_image, mask=cv2.bitwise_not(mask_image))

    # Concatenate the images side-by-side
    composite = np.hstack((overlay_image, actual_image, mask_removed, masked_image))

    # Resize the composite image for better visualization
    composite_resized = cv2.resize(composite, 
                                   (composite.shape[1] // 2, composite.shape[0] // 2))

    # Save the composite image
    cv2.imwrite(output_path, composite_resized)

    return composite_resized

def black_mask(image_path, mask_path, output_path):
    actual_image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_removed = cv2.bitwise_and(actual_image, actual_image, mask=cv2.bitwise_not(mask_image))
    resized = cv2.resize(mask_removed, 
                                   (mask_removed.shape[1] // 2, mask_removed.shape[0] // 2))
    cv2.imwrite(output_path, resized)

def closing(mask_path, output_path):
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(output_path, closing)



# Paths
image_dir = "fiverr/data/images/"  # Image directory with .jpg files
mask_dir = "fiverr/data/masks/"    # Mask directory with .png files
output_dir = "fiverr/output/closed" # Output directory

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all images
for filename in tqdm(os.listdir(image_dir)):
    if filename.endswith(".JPG"):
        image_file = os.path.join(image_dir, filename)
        mask_file = os.path.join(mask_dir, filename.replace(".JPG", ".png"))
        output_file = os.path.join(output_dir, filename.replace(".JPG", ".png"))
        try:
            # composite_image(image_file, mask_file, output_file, alpha=0.8)
            # black_mask(image_file, mask_file, output_file)
            closing(mask_file, output_file)
        except FileNotFoundError as e:
            print(e)
