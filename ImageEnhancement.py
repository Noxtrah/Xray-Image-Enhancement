import os
import json
import cv2
import numpy as np
import supervisely as sly
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define paths
dataset_dir = r"D:\Teeth Segmentation Dataset"
output_dir = "./processed_results"
filtered_original_dir = "./filtered_original_results"
filtered_dir = "./filtered_results"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(filtered_original_dir, exist_ok=True)
os.makedirs(filtered_dir, exist_ok=True)

# Create Supervisely project from local directory
project = sly.Project(dataset_dir, sly.OpenMode.READ)
print("Opened project: ", project.name)
print("Number of images in project:", project.total_items)

# Display annotations tags and classes
print(project.meta)
for obj_class in project.meta.obj_classes:
    print(f"Class '{obj_class.name}': geometry='{obj_class.geometry_type}', color='{obj_class.color}'")
for tag in project.meta.tag_metas:
    print(f"Tag '{tag.name}': color='{tag.color}'")

print("Number of datasets (folders) in project:", len(project.datasets))

# Helper functions for applying filters
def apply_contrast_enhancement(img):
    """Apply contrast enhancement using CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def apply_median_filter(img):
    """Apply a median filter to the image."""
    return cv2.medianBlur(img, 5)

def apply_sharpening_filter(img):
    """Apply sharpening filter to the image."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def save_filtered_image(original_img, contrast, median, sharpened, folder, index, item_name):
    """Save original and filtered images stacked horizontally and as a preview."""
    # Save the stacked image
    stacked_image = np.hstack([original_img, contrast, median, sharpened])
    filtered_image_path = os.path.join(folder, f"filtered_{index}_{item_name}.jpg")
    cv2.imwrite(filtered_image_path, stacked_image)

    # Create a preview with matplotlib
    plt.figure(figsize=(12, 8))
    images = [original_img, contrast, median, sharpened]
    titles = ['Original', 'Contrast Enhanced', 'Median Filtered', 'Sharpened']
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    preview_path = os.path.join(folder, f"filtered_preview_{index}_{item_name}.jpg")
    plt.savefig(preview_path)
    plt.close()

# Process datasets and save results
counter = 0
progress = tqdm(project.datasets, desc="Processing datasets")
for dataset in project.datasets:
    for item_name, image_path, ann_path in dataset.items():
        if counter >= 5:
            break

        print(f"Processing '{item_name}': image='{image_path}', ann='{ann_path}'")

        # Load annotations
        with open(ann_path, 'r') as f:
            ann_json = json.load(f)
        ann = sly.Annotation.from_json(ann_json, project.meta)

        # Load original image
        original_img = sly.image.read(image_path)

        # Draw annotations
        annotated_img = original_img.copy()
        for label in ann.labels:
            label.draw(annotated_img)

        # Save annotated image
        annotated_path = os.path.join(output_dir, f"annot_{counter + 1}_{item_name}.jpg")
        sly.image.write(annotated_path, annotated_img)

        # Apply and save filters for original and annotated images
        for img, folder in [(original_img, filtered_original_dir), (annotated_img, filtered_dir)]:
            contrast = apply_contrast_enhancement(img)
            median = apply_median_filter(img)
            sharpened = apply_sharpening_filter(img)
            save_filtered_image(img, contrast, median, sharpened, folder, counter + 1, item_name)

        counter += 1
        if counter >= 5:
            break
    if counter >= 5:
        break
