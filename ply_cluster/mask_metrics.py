import numpy as np
import os
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_metrics(gt_mask, pred_mask):
    """Calculate segmentation metrics: IoU, Precision, Recall, F1-score, and FPR."""
    gt_mask = gt_mask / 255  # Ensure binary (0, 1)
    pred_mask = pred_mask / 255  # Ensure binary (0, 1)

    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    tp = intersection
    fp = (pred_mask - gt_mask).clip(min=0).sum()
    fn = (gt_mask - pred_mask).clip(min=0).sum()
    tn = np.logical_and(~gt_mask.astype(bool), ~pred_mask.astype(bool)).sum()

    iou = tp / union if union > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "FPR": fpr
    }

masks_dir = "ply_cluster/output/kitchen/masks/"  # Image directory with .jpg files
gt_dir = "ply_cluster/data/kitchen/masks_gt/"    # Mask directory with .png files

metrics = {
    "IoU": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "FPR": []
}

def process_file(filename, masks_dir, gt_dir):
    """Process a single file and calculate metrics."""
    gt_file = os.path.join(gt_dir, filename)
    mask_file = os.path.join(masks_dir, filename)
    try:
        gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None or pred_mask is None:
            raise FileNotFoundError(f"File not found: {filename}")
        return calculate_metrics(gt_mask, pred_mask)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None
    


# Directories
masks_dir = "ply_cluster/output/kitchen/masks/"
gt_dir = "ply_cluster/data/kitchen/masks_gt/"

# Collect files
files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

# Multithreading
metrics = {
    "IoU": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "FPR": []
}

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_file, f, masks_dir, gt_dir): f for f in files}
    for future in tqdm(as_completed(futures), total=len(files), desc="Processing"):
        result = future.result()
        if result:
            for key, value in result.items():
                metrics[key].append(value)

# Calculate average metrics
average_metrics = {key: np.mean(values) for key, values in metrics.items()}
print("Average Metrics:")
print(average_metrics)