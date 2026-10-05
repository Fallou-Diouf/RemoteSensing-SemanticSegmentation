import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score

# Allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import VaihingenDataset
from model import Hypercolumns


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
IMAGE_FOLDER = os.path.join(DATA_FOLDER, "top")
GT_FOLDER = os.path.join(DATA_FOLDER, "1CGT")

WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT,
    "weights",
    "Hypercolumns_2epochs.pth"
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 3
NUM_CLASSES = 6


# ============================================================
# Device
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# Test dataset
# ============================================================

test_dataset = VaihingenDataset(
    IMAGE_FOLDER,
    GT_FOLDER,
    "test"
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test patches:", len(test_dataset))


# ============================================================
# Model
# ============================================================

network = Hypercolumns()

network.load_state_dict(
    torch.load(
        WEIGHTS_PATH,
        map_location=device
    )
)

network.to(device)
network.eval()


# ============================================================
# Evaluation
# ============================================================

all_predictions = []
all_ground_truths = []

with torch.no_grad():

    for inputs, GTs in test_loader:

        inputs = inputs.to(device)

        predictions = network(inputs)

        predicted_classes = torch.argmax(
            predictions,
            dim=1
        )

        all_predictions.append(
            predicted_classes.cpu().numpy()
        )

        all_ground_truths.append(
            GTs.numpy()
        )


# ============================================================
# Convert to NumPy arrays
# ============================================================

all_predictions = np.concatenate(
    all_predictions,
    axis=0
)

all_ground_truths = np.concatenate(
    all_ground_truths,
    axis=0
)


# ============================================================
# Accuracy
# ============================================================

accuracy = accuracy_score(
    all_ground_truths.flatten(),
    all_predictions.flatten()
)

print(f"\nPixel accuracy: {accuracy:.4f}")


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_ground_truths.flatten(),
    all_predictions.flatten(),
    labels=np.arange(NUM_CLASSES)
)

print("\nConfusion matrix:")
print(cm)


# ============================================================
# IoU for each class
# ============================================================

ious = []

for class_id in range(NUM_CLASSES):
    true_positive = cm[class_id, class_id]
    false_positive = cm[:, class_id].sum() - true_positive
    false_negative = cm[class_id, :].sum() - true_positive

    iou = true_positive / (true_positive + false_positive + false_negative)
    ious.append(iou)

# Mean IoU
mean_iou = np.mean(ious)

print("\nIoU per class:")
for class_id, iou in enumerate(ious):
    print(f"Class {class_id}: {iou:.4f}")

print(f"\nmIoU: {mean_iou:.4f}")