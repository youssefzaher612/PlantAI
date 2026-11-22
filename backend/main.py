
import io
import base64
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "faba_best.pth"
SAM_CHECKPOINT = BASE_DIR / "models" / "sam_vit_b_01ec64.pth"

FRONTEND_DIR = BASE_DIR / "frontend"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "faba_blight",
    "faba_chocolate",
    "faba_healthy",
    "faba_rust"
]

CLASS_DISPLAY_NAMES = {
    "faba_blight": "Blight",
    "faba_chocolate": "Chocolate Spot",
    "faba_healthy": "Healthy",
    "faba_rust": "Rust"
}

IMG_SIZE = 224
FILL = (200, 200, 200)

CONF_MIN = 0.80
MARGIN_MIN = 0.20

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# =========================================================
# 2. LOAD RESNET18 CLASSIFICATION MODEL
# =========================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ResNet checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    classes = checkpoint.get("classes", CLASS_NAMES)
    img_size = checkpoint.get("img_size", IMG_SIZE)

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            model.fc.in_features,
            len(classes)
        )
    )

    state_dict = checkpoint.get("model", checkpoint)

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    print("ResNet18 loaded successfully.")
    print("Classes:", classes)
    print("Image size:", img_size)

    return model, classes, img_size


model, classes, IMG_SIZE = load_model()


# =========================================================
# 3. LOAD SAM SEGMENTATION MODEL
# =========================================================

def load_sam():

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {SAM_CHECKPOINT}"
        )

    sam = sam_model_registry["vit_b"](
        checkpoint=str(SAM_CHECKPOINT)
    )

    sam = sam.to(DEVICE)
    sam.eval()

    generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=24,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.90,
    min_mask_region_area=300,
    crop_n_layers=0
)

    print("SAM loaded successfully on", DEVICE)

    return generator


mask_generator = load_sam()


# =========================================================
# 4. IMAGE PREPROCESSING
# =========================================================

def to_model_input(rgba, size=224, fill=FILL):

    rgb = rgba[..., :3]

    alpha = rgba[..., 3:] / 255.0

    img = (
        rgb * alpha +
        np.array(fill) * (1 - alpha)
    ).astype(np.uint8)

    h, w = img.shape[:2]

    side = max(h, w)

    canvas = np.empty(
        (side, side, 3),
        dtype=np.uint8
    )

    canvas[:] = np.array(fill, dtype=np.uint8)

    y_offset = (side - h) // 2
    x_offset = (side - w) // 2

    canvas[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = img

    resized = cv2.resize(
        canvas,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    return resized


def to_tensor(image):

    x = torch.from_numpy(image)

    x = x.to(DEVICE)

    x = x.permute(2, 0, 1)

    x = x.unsqueeze(0)

    x = x.float() / 255.0

    mean = torch.tensor(
        MEAN,
        device=DEVICE
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        STD,
        device=DEVICE
    ).view(1, 3, 1, 1)

    return (x - mean) / std


# =========================================================
# 5. SAM LEAF MASK DETECTION AND FILTERING
# =========================================================





def find_leaf_masks(image):

    H, W = image.shape[:2]
    image_area = H * W

    proposals = mask_generator.generate(image)

    candidates = []

    for p in proposals:

        mask = p["segmentation"]
        area = int(mask.sum())

        # 1. Reject very small regions
        if area < 0.025 * image_area:
            continue

        # 2. Reject masks covering too much of the image
        if area > 0.65 * image_area:
            continue

        pixels = image[mask].astype(np.float32)

        if len(pixels) == 0:
            continue

        # 3. Green ratio
        green_ratio = np.mean(
            (pixels[:, 1] > pixels[:, 0] * 0.80) &
            (pixels[:, 1] > pixels[:, 2] * 0.80)
        )

        if green_ratio < 0.35:
            continue

        # 4. SAM quality filtering
        predicted_iou = float(p.get("predicted_iou", 0))
        stability = float(p.get("stability_score", 0))

        if predicted_iou < 0.86:
            continue

        if stability < 0.90:
            continue

        # 5. Bounding box
        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        box_width = x_max - x_min + 1
        box_height = y_max - y_min + 1

        aspect_ratio = box_width / max(box_height, 1)

        if aspect_ratio < 0.20 or aspect_ratio > 4.0:
            continue

        # 6. Candidate score
        score = (
            0.4 * predicted_iou +
            0.4 * stability +
            0.2 * green_ratio
        )

        candidates.append({
            "mask": mask,
            "area": area,
            "score": score,
            "green_ratio": float(green_ratio)
        })

    # IMPORTANT:
    # Process larger masks first so that small fragments
    # inside a complete leaf are rejected.
    candidates.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    selected = []

    for candidate in candidates:

        mask = candidate["mask"]

        duplicate = False

        for old in selected:

            old_mask = old["mask"]

            intersection = np.logical_and(
                mask,
                old_mask
            ).sum()

            smaller_area = min(
                candidate["area"],
                old["area"]
            )

            if smaller_area == 0:
                continue

            overlap_ratio = intersection / smaller_area

            # Reject duplicate or contained fragments
            if overlap_ratio > 0.65:
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)

    # Return multiple detected leaves
    # Largest valid regions first
    return selected[:10]


# =========================================================
# 6. CLASSIFY ONE DETECTED LEAF
# =========================================================

def diagnose_leaf(image, mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    y0 = ys.min()
    y1 = ys.max() + 1

    x0 = xs.min()
    x1 = xs.max() + 1

    crop = image[y0:y1, x0:x1].copy()

    crop_mask = mask[y0:y1, x0:x1]

    rgba = np.dstack([
        crop,
        crop_mask.astype(np.uint8) * 255
    ])

    # Apply notebook preprocessing
    model_input = to_model_input(
        rgba,
        IMG_SIZE,
        FILL
    )

    x = to_tensor(model_input)

    model.eval()

    with torch.inference_mode():

        logits = model(x).float()

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

    order = np.argsort(probabilities)[::-1]

    top = int(order[0])
    second = int(order[1])

    confidence = float(probabilities[top])

    second_confidence = float(
        probabilities[second]
    )

    margin = confidence - second_confidence

    predicted_class = classes[top]
    second_class = classes[second]

    prediction = CLASS_DISPLAY_NAMES.get(
        predicted_class,
        predicted_class
    )

    second_prediction = CLASS_DISPLAY_NAMES.get(
        second_class,
        second_class
    )

    uncertain = (
        confidence < CONF_MIN or
        margin < MARGIN_MIN
    )

    return {
        "crop": rgba,
        "model_input": model_input,
        "prediction": prediction,
        "confidence": confidence,
        "second_prediction": second_prediction,
        "second_confidence": second_confidence,
        "uncertain": uncertain
    }


def encode_rgba_image(image):

    # Convert RGBA to BGRA for OpenCV PNG encoding
    bgra = cv2.cvtColor(
        image,
        cv2.COLOR_RGBA2BGRA
    )

    success, buffer = cv2.imencode(
        ".png",
        bgra
    )

    if not success:
        raise ValueError("Could not encode transparent image.")

    return base64.b64encode(
        buffer
    ).decode("utf-8")


# =========================================================
# 7. IMAGE ENCODING FOR FRONTEND
# =========================================================

def encode_image(image):

    success, buffer = cv2.imencode(
        ".png",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

    if not success:
        raise ValueError("Could not encode image.")

    return base64.b64encode(
        buffer
    ).decode("utf-8")


# =========================================================
# 8. DRAW SEGMENTATION RESULTS
# =========================================================

def draw_leaf_results(image, masks):

    output = image.copy()

    for index, item in enumerate(masks, start=1):

        mask = item["mask"]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            output,
            contours,
            -1,
            (255, 0, 0),
            3
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        center_x = int((xs.min() + xs.max()) / 2)
        center_y = int((ys.min() + ys.max()) / 2)

        cv2.putText(
            output,
            str(index),
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )

    return output


# =========================================================
# 9. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="FabaCare API",
    description="Faba Bean Leaf Disease Detection",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# 10. HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():

    return {
        "status": "running",
        "model_loaded": model is not None,
        "sam_loaded": mask_generator is not None,
        "device": str(DEVICE),
        "classes": list(classes)
    }


# =========================================================
# 11. PREDICTION ENDPOINT
# =========================================================

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WEBP image."
        )

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        pil_image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image = np.array(pil_image)

    except HTTPException:
        raise

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )

    # Generate and filter SAM masks
    leaves = find_leaf_masks(image)

    if len(leaves) == 0:

        return {
            "success": False,
            "message": (
                "No leaf regions were detected. "
                "Please upload a clearer faba bean leaf image."
            ),
            "total_leaves": 0,
            "results": []
        }

    results = []

    # Classify every detected leaf
    for index, leaf in enumerate(leaves, start=1):

        diagnosis = diagnose_leaf(
            image,
            leaf["mask"]
        )

        if diagnosis is None:
            continue

        results.append({
            "leaf_number": index,
            "prediction": diagnosis["prediction"],
            "confidence": diagnosis["confidence"],
            "second_prediction": diagnosis["second_prediction"],
            "second_confidence": diagnosis["second_confidence"],
            "uncertain": diagnosis["uncertain"],
            "crop_image": encode_rgba_image(diagnosis["crop"]),
            "model_input_image": encode_image(
                diagnosis["model_input"]
            )
        })

    # Draw all detected leaf contours
    segmented_image = draw_leaf_results(
        image,
        leaves
    )

    # Backward-compatible top result fields
    first_result = results[0] if results else None

    return {
        "success": True,
        "total_leaves": len(results),

        "prediction": (
            first_result["prediction"]
            if first_result else None
        ),

        "confidence": (
            first_result["confidence"]
            if first_result else None
        ),

        "second_prediction": (
            first_result["second_prediction"]
            if first_result else None
        ),

        "second_confidence": (
            first_result["second_confidence"]
            if first_result else None
        ),

        "uncertain": (
            first_result["uncertain"]
            if first_result else True
        ),

        "segmented_image": encode_image(
            segmented_image
        ),

        "results": results
    }


# =========================================================
# 12. SERVE FRONTEND
# =========================================================

if FRONTEND_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static"
    )


@app.get("/")
def home():

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found."
        )

    return FileResponse(index_file)
