import io, zipfile, random, shutil, datetime
from pathlib import Path
from PIL import Image
from .config import DATA_DIR, VAL_RATIO

def unpack_zip_and_split(zip_bytes: bytes, val_ratio: float = VAL_RATIO):
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(DATA_DIR / "raw")

    refresh_train_val_split(val_ratio)

def save_single_image(label: str, img_bytes: bytes) -> str:
    cls_dir = DATA_DIR / "raw" / label
    cls_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    fp = cls_dir / f"{ts}.jpg"
    Image.open(io.BytesIO(img_bytes)).convert("RGB").save(fp, "JPEG", quality=95)
    return str(fp)

def refresh_train_val_split(val_ratio: float = VAL_RATIO):
    for subset in ("train", "val"):
        d = DATA_DIR / subset
        if d.exists():
            shutil.rmtree(d)

    for cls_dir in (DATA_DIR / "raw").iterdir():
        if not cls_dir.is_dir():
            continue
        imgs = list(cls_dir.glob("*"))
        if not imgs:
            continue

        random.shuffle(imgs)
        cut = max(1, int(len(imgs) * val_ratio))  # Ensure at least 1 goes to val

        val_imgs = imgs[:cut]
        train_imgs = imgs[cut:]

        for subset, lst in [("train", train_imgs), ("val", val_imgs)]:
            out_cls = DATA_DIR / subset / cls_dir.name
            out_cls.mkdir(parents=True, exist_ok=True)
            for im in lst:
                shutil.copy2(im, out_cls / im.name)
