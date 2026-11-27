
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "mobilenet_custom.pth"
LABELS_PATH = BASE_DIR / "class_names.txt"

# ==========================================
# 2. TRAINING CONFIGURATION
# ==========================================

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

# ==========================================
# 3. IMAGE TRANSFORMATIONS
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 4. LOAD DATASET
# ==========================================

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

dataset = datasets.ImageFolder(
    root=str(DATA_DIR),
    transform=transform
)

class_labels = dataset.classes
num_classes = len(class_labels)

if num_classes != 4:
    raise ValueError(
        f"Expected 4 classes, but found {num_classes}: {class_labels}"
    )

print("\nDetected classes:", class_labels)
print("Total images:", len(dataset))

if len(dataset) == 0:
    raise ValueError("No images found in the dataset.")

# ==========================================
# 5. TRAIN / VALIDATION SPLIT
# ==========================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", train_size)
print("Validation images:", val_size)

# ==========================================
# 6. LOAD PRETRAINED MOBILENETV2
# ==========================================

print("\nLoading pretrained MobileNetV2...")

weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)

# Freeze feature extraction layers initially
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Fine-tune the last two feature blocks
for parameter in model.features[-2:].parameters():
    parameter.requires_grad = True

# Replace final classifier for our four classes
model.classifier[1] = nn.Linear(
    model.last_channel,
    num_classes
)

# ==========================================
# 7. DEVICE, LOSS AND OPTIMIZER
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

print("\nTraining device:", device)

# ==========================================
# 8. MODEL TRAINING AND VALIDATION
# ==========================================

print("\nStarting model training...\n")

for epoch in range(EPOCHS):

    # -------- TRAINING --------

    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total

    # -------- VALIDATION --------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

# ==========================================
# 9. SAVE TRAINED MODEL
# ==========================================

torch.save(model.state_dict(), MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)

# ==========================================
# 10. SAVE CLASS LABELS
# ==========================================

with open(LABELS_PATH, "w", encoding="utf-8") as file:
    for label in class_labels:
        file.write(label + "\n")

print("\nClass labels saved successfully:")
print(LABELS_PATH)

print("\n========== TRAINING COMPLETED ==========")
