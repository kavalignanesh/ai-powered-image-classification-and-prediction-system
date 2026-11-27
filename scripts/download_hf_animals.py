
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import hashlib
import io

# ---------------- SETTINGS ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TARGET_PER_CLASS = 500
IMAGE_SIZE = (224, 224)

CLASSES = {
    1: "cat",
    4: "dog",
    6: "horse"
}

# -------------------------------------------

counts = {}
hashes = set()

# Create folders and detect previously downloaded images
for animal in CLASSES.values():
    folder = DATASET_DIR / animal
    folder.mkdir(parents=True, exist_ok=True)

    existing_images = list(folder.glob("*.jpg"))
    counts[animal] = len(existing_images)

    # Load hashes of existing images for duplicate checking
    for path in existing_images:
        try:
            hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            print(f"Could not read: {path}")

print("\nExisting image counts:", counts)

if all(count >= TARGET_PER_CLASS for count in counts.values()):
    print("All requested images are already available!")
    raise SystemExit

print("\nConnecting to Hugging Face...")

dataset = load_dataset(
    "Rapidata/Animals-10",
    split="train",
    streaming=True
)

print("Starting image collection...\n")

try:
    for row in dataset:

        label = row["label"]

        # Ignore classes we don't need
        if label not in CLASSES:
            continue

        animal = CLASSES[label]

        # Skip classes that already reached the target
        if counts[animal] >= TARGET_PER_CLASS:
            if all(v >= TARGET_PER_CLASS for v in counts.values()):
                break
            continue

        try:
            image = row["image"].convert("RGB")
            image = image.resize(IMAGE_SIZE)

            # Prepare image bytes
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            image_bytes = buffer.getvalue()

            # Check duplicates
            digest = hashlib.sha256(image_bytes).hexdigest()

            if digest in hashes:
                continue

            # Save image
            next_number = counts[animal] + 1
            filename = f"{animal}_{next_number:03d}.jpg"
            save_path = DATASET_DIR / animal / filename

            save_path.write_bytes(image_bytes)

            hashes.add(digest)
            counts[animal] += 1

            print(
                f"{animal}: "
                f"{counts[animal]}/{TARGET_PER_CLASS}"
            )

        except Exception as image_error:
            print(f"Skipped invalid image: {image_error}")
            continue

except KeyboardInterrupt:
    print("\nDownload interrupted by user.")

except Exception as error:
    print(f"\nDownload stopped due to an error: {error}")

finally:
    print("\n========== DOWNLOAD SUMMARY ==========")
    for animal, count in counts.items():
        print(f"{animal}: {count}/{TARGET_PER_CLASS}")

    print("======================================")
