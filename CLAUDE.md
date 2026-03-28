# CLAUDE.md

## Project overview

Fork of [DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2) — a 3D mesh generation pipeline using sparse spatial attention. Generates 3D meshes from single images via a multi-stage pipeline: dense inference → sparse 512 → (optional) sparse 1024 → refiner.

## Docker

- **Build**: `docker build -t direct3d-s2 -f docker/Dockerfile .` (run from repo root)
- **Run**: `docker run -it --gpus all -p 7860:7860 -v ~/direct3d-outputs:/workspace/outputs direct3d-s2`
- Dockerfile uses `COPY` (not git clone) — code changes are detected automatically, no `--no-cache` needed
- Compilation-heavy layers (flash-attn, voxelize, torchsparse) are cached above the `COPY .` layer
- Model weights (Direct3D-S2, BiRefNet, DINOv2) are baked into the image
- torchsparse requires `FORCE_CUDA=1` during docker build (no GPU runtime available, `torch.cuda.is_available()` returns False)
- torchsparse source tree must be deleted after install or it shadows the compiled `.so` backend (CPU-only `__init__.py` vs CUDA extension)

## App flags

- `python app.py` — default, loads all models
- `python app.py --low-vram` — skips 1024 models and refiner (~14GB VRAM needed by refiner alone), 512-only output
- `python app.py --share` — enables public Gradio link (needed for RunPod)

## Architecture notes

- Pipeline stages run sequentially; models are offloaded to CPU between stages to free VRAM
- The `direct3d_s2/` package requires an `__init__.py` (was missing upstream) for `find_packages()` to work
- `sparse_image_encoder` is shared between 512 and 1024 stages
- Generated shell scripts in the container need explicit `source conda.sh && conda activate` (non-interactive shells don't source `.bashrc`)

## VRAM requirements

- 512 without refiner (--low-vram): ~10GB
- 512 with refiner: ~16-20GB
- 1024 with refiner: ~24GB+
