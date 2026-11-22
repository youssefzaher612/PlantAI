# PlantAI

This repository contains FabaCare, a web application for detecting faba bean leaf diseases. A FastAPI backend uses a trained ResNet18 classifier and Segment Anything (SAM, ViT-B) to segment candidate leaf regions before classification. The frontend supports image upload and displays diagnosis, confidence, alternative predictions, and segmentation results.

## Current status

The end-to-end faba bean workflow is connected. Segmentation consistency and individual-leaf presentation need further evaluation. Corn support is planned for a later integration with a separate module and is not implemented yet.

## Requirements

- Python 3.10 or newer
- Git LFS for model checkpoint files
- Enough memory/compute to load SAM ViT-B (CUDA is used when available; otherwise CPU)

## Setup

```powershell
git lfs install
git lfs pull
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The backend also exposes `/api/health` and accepts image predictions at `/api/predict`.

The application expects these checkpoints at their existing paths:

- `models/faba_best.pth` - trained FabaCare ResNet18 checkpoint
- `models/sam_vit_b_01ec64.pth` - SAM ViT-B checkpoint

Both files are tracked with Git LFS. A Git LFS quota or bandwidth limit on the GitHub account may affect cloning or downloading the SAM checkpoint.

## Disease classes

- Healthy
- Blight
- Rust
- Chocolate Spot

## Project tasks

The next-week plan from the progress report is recorded in [TASKS.md](TASKS.md). It covers segmentation refinement, individual-leaf presentation, the proposed corn integration, and conditional integration testing.

## Notes

- The backend loads both model checkpoints when the application starts.
- Results depend on image quality, background, lighting, leaf overlap, and SAM mask quality.
- Current testing is preliminary; broader validation on representative images is still needed.
