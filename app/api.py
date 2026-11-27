from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from .data_utils import (
    unpack_zip_and_split,
    save_single_image,
    refresh_train_val_split,
)
from .model_utils import train_model, predict, load_model, MODEL, LABELS

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/")
def index():
    return {"msg": "Go to /web or use /train /add_image /predict endpoints"}

@router.get("/web", response_class=HTMLResponse)
def web_ui(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={"request": request}
)
@router.post("/predict_form", response_class=HTMLResponse)
async def form_predict(request: Request, image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        label, conf = predict(img_bytes)

        result = {
            "prediction": label,
            "confidence": f"{conf*100:.2f}%"
        }

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request": request, "result": result}
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "result": {
                    "prediction": "Error",
                    "confidence": str(e)
                }
            }
        )

@router.post("/add_image")
async def add_image(label: str = Form(...), images: List[UploadFile] = File(...)):
    saved = []
    for img in images:
        if not img.content_type.startswith("image/"):
            raise HTTPException(400, f"{img.filename} is not an image.")
        saved.append(save_single_image(label, await img.read()))
    refresh_train_val_split()
    return {"status": "added", "label": label, "files": saved}

@router.post("/train")
async def train(dataset: UploadFile = File(None)):
    if dataset is not None:
        zip_bytes = await dataset.read()
        unpack_zip_and_split(zip_bytes)
    else:
        refresh_train_val_split()
    classes = train_model()
    global MODEL, LABELS
    MODEL, LABELS = load_model()
    return {"status": "trained", "classes": classes}

@router.post("/predict")
async def api_predict(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        label, conf = predict(img_bytes)
        return {
            "filename": image.filename,
            "prediction": label,
            "confidence": f"{conf*100:.2f}%"
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
