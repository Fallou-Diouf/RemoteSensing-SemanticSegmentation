import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import VaihingenDataset
from model import Hypercolumns


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
IMAGE_FOLDER = os.path.join(DATA_FOLDER, "top")
GT_FOLDER = os.path.join(DATA_FOLDER, "1CGT")

WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT,
    "weights",
    "Hypercolumns_10epochs.pth"
)

FIGURES_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

os.makedirs(FIGURES_FOLDER, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

test_dataset = VaihingenDataset(
    IMAGE_FOLDER,
    GT_FOLDER,
    "test"
)

print("Test patches:", len(test_dataset))


# --------------------------------------------------
# Model
# --------------------------------------------------

network = Hypercolumns()

network.load_state_dict(
    torch.load(
        WEIGHTS_PATH,
        map_location=device
    )
)

network.to(device)
network.eval()


# --------------------------------------------------
# Select one patch
# --------------------------------------------------

index = 0

image, ground_truth = test_dataset[index]

input_tensor = image.unsqueeze(0).to(device)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    output = network(input_tensor)

    prediction = torch.argmax(
        output,
        dim=1
    )

prediction = prediction.squeeze(0).cpu().numpy()

image = image.permute(1, 2, 0).numpy()
ground_truth = ground_truth.numpy()


# --------------------------------------------------
# Visualization
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Input image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ground_truth)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(prediction)
plt.title("Prediction")
plt.axis("off")

plt.tight_layout()

output_path = os.path.join(
    FIGURES_FOLDER,
    "segmentation_prediction.png"
)

plt.savefig(output_path)
plt.show()

print("Figure saved to:", output_path)