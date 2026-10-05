import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import VaihingenDataset
from model import Hypercolumns


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
IMAGE_FOLDER = os.path.join(DATA_FOLDER, "top")
GT_FOLDER = os.path.join(DATA_FOLDER, "1CGT")

WEIGHTS_FOLDER = os.path.join(PROJECT_ROOT, "weights")
FIGURES_FOLDER = os.path.join(PROJECT_ROOT, "results", "figures")

os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)


BATCH_SIZE = 3
NUMBER_EPOCHS = 2
LEARNING_RATE = 0.001


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
# Dataset
# ============================================================

training_dataset = VaihingenDataset(
    IMAGE_FOLDER,
    GT_FOLDER,
    "train"
)

validate_dataset = VaihingenDataset(
    IMAGE_FOLDER,
    GT_FOLDER,
    "val"
)


train_loader = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validate_loader = torch.utils.data.DataLoader(
    dataset=validate_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training patches:", len(training_dataset))
print("Validation patches:", len(validate_dataset))


# ============================================================
# Model
# ============================================================

network = Hypercolumns()
network.to(device)


# ============================================================
# Loss and optimizer
# ============================================================

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    network.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

training_losses = []
validation_losses = []


for epoch in range(NUMBER_EPOCHS):

    print(f"\nStarting epoch {epoch + 1}/{NUMBER_EPOCHS}")

    # -------------------------
    # Training
    # -------------------------

    network.train()

    training_loss = 0.0

    for inputs, GTs in tqdm(train_loader, desc="Training"):

        inputs = inputs.to(device)

        GTs = GTs.long().to(device)

        optimizer.zero_grad()

        predictions = network(inputs)

        loss = loss_function(predictions, GTs)

        loss.backward()

        optimizer.step()

        training_loss += loss.item()

    training_loss /= len(train_loader)

    training_losses.append(training_loss)

    print(f"Training loss: {training_loss:.4f}")


    # -------------------------
    # Validation
    # -------------------------

    network.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for inputs, GTs in tqdm(validate_loader, desc="Validation"):

            inputs = inputs.to(device)

            GTs = GTs.long().to(device)

            predictions = network(inputs)

            loss = loss_function(predictions, GTs)

            validation_loss += loss.item()

    validation_loss /= len(validate_loader)

    validation_losses.append(validation_loss)

    print(f"Validation loss: {validation_loss:.4f}")


# ============================================================
# Loss curve
# ============================================================

plt.figure()

plt.plot(
    np.arange(1, NUMBER_EPOCHS + 1),
    training_losses,
    label="Training loss"
)

plt.plot(
    np.arange(1, NUMBER_EPOCHS + 1),
    validation_losses,
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(
    os.path.join(FIGURES_FOLDER, "training_validation_loss.png")
)

plt.close()


# ============================================================
# Save model weights
# ============================================================

weights_path = os.path.join(
    WEIGHTS_FOLDER,
    f"Hypercolumns_{NUMBER_EPOCHS}epochs.pth"
)

torch.save(network.state_dict(), weights_path)

print("\nTraining finished.")
print("Weights saved to:", weights_path)