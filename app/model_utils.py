
import copy
import io

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from .config import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Minimum confidence required for a valid prediction
CONFIDENCE_THRESHOLD = 0.70


# 1. Image preprocessing
def _transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


# 2. Load training and validation datasets
def _get_loaders():

    train_path = DATA_DIR / "train"
    val_path = DATA_DIR / "val"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Training data not found. Prepare data/raw first."
        )

    transform = _transforms()

    train_ds = datasets.ImageFolder(
        train_path,
        transform=transform
    )

    val_ds = datasets.ImageFolder(
        val_path,
        transform=transform
    )

    if train_ds.classes != val_ds.classes:
        raise ValueError("Training and validation classes do not match.")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_ds, val_ds, train_loader, val_loader


# 3. Initialize MobileNetV2
def _init_model(num_classes):

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    return model.to(DEVICE)


# 4. Train the model
def train_model():

    train_ds, val_ds, train_loader, val_loader = _get_loaders()

    classes = train_ds.classes
    num_classes = len(classes)

    if num_classes != 4:
        raise ValueError(f"Expected 4 classes, found {classes}")

    model = _init_model(num_classes)

    # Load previously trained MobileNetV2 weights
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    print("Loaded existing trained MobileNetV2 model.")

    # Fine-tune the final feature blocks and classifier
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # Training and validation
    for epoch in range(EPOCHS):

        for phase, loader in [
            ("train", train_loader),
            ("val", val_loader)
        ]:

            model.train(phase == "train")

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                predictions = outputs.argmax(dim=1)

                running_loss += loss.item() * images.size(0)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_accuracy = correct / total

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"{phase} Loss: {epoch_loss:.4f} | "
                f"{phase} Accuracy: {epoch_accuracy * 100:.2f}%"
            )

            if phase == "val" and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_weights = copy.deepcopy(model.state_dict())

    # Save best model
    model.load_state_dict(best_weights)

    torch.save(model.state_dict(), MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        file.writelines(label + "\n" for label in classes)

    print("Best validation accuracy:", best_acc * 100)
    print("Model saved:", MODEL_PATH)

    return classes


# 5. Load trained model
def load_model():

    with open(LABELS_PATH, encoding="utf-8") as file:
        labels = [line.strip() for line in file if line.strip()]

    model = _init_model(len(labels))

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    return model, labels


# 6. Model initialization
MODEL, LABELS = None, None


def load_model_if_needed():

    global MODEL, LABELS

    if MODEL is None or LABELS is None:

        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            raise ValueError("Trained model or class labels not found.")

        MODEL, LABELS = load_model()


# 7. Image prediction with invalid-input threshold
def predict(img_bytes: bytes):

    load_model_if_needed()

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensor = _transforms()(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = MODEL(tensor)
        probabilities = torch.softmax(output[0], dim=0)

    index = probabilities.argmax().item()
    confidence = float(probabilities[index])

    # Reject predictions below 70% confidence
    if confidence < CONFIDENCE_THRESHOLD:
        return "Invalid Input", 0.0

    return LABELS[index], confidence
