
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import hashlib
import io

# ---------------- SETTINGS ----------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
ANIMAL = "tiger"
TARGET = 500
IMAGE_SIZE = (224, 224)
TIGER_LABEL = 292

# -------------------------------------------

folder = DATASET_DIR / ANIMAL
folder.mkdir(parents=True, exist_ok=True)

# Count existing images and detect duplicates
existing_images = list(folder.glob("*.jpg"))
count = len(existing_images)

hashes = set()

for path in existing_images:
    try:
        hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    except OSError:
        print(f"Could not read: {path}")

print(f"Existing tiger images: {count}/{TARGET}")

if count >= TARGET:
    print("Target already reached!")
    raise SystemExit

print("\nConnecting to Hugging Face...")

dataset = load_dataset(
    "mlnomad/imnet1k_tiger_Panthera_tigris",
    split="train",
    streaming=True
)

print("Starting tiger image collection...\n")

try:
    for row in dataset:

        if row["label"] != TIGER_LABEL:
            continue

        if count >= TARGET:
            break

        try:
            image = row["image"].convert("RGB")
            image = image.resize(IMAGE_SIZE)

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            image_bytes = buffer.getvalue()

            digest = hashlib.sha256(image_bytes).hexdigest()

            if digest in hashes:
                continue

            count += 1
            filename = f"tiger_{count:03d}.jpg"
            save_path = folder / filename

            save_path.write_bytes(image_bytes)
            hashes.add(digest)

            print(f"Tiger: {count}/{TARGET}")

        except Exception as image_error:
            print(f"Skipped image: {image_error}")
            continue

except KeyboardInterrupt:
    print("\nDownload interrupted.")

except Exception as error:
    print(f"\nDownload stopped: {error}")

finally:
    print("\n========== DOWNLOAD SUMMARY ==========")
    print(f"Tiger: {count}/{TARGET}")
    print("======================================")
