# SRFD-DETR

Scale-Robust Feature Disentanglement Transformer for cross-scale agglomeration detection (RT-DETR–based implementation). This repository couples **MEG** (micro-edge-gray, diffusion-based preprocessing) with **SRFD** (domain-invariant / domain-specific feature streams and losses).

> **License notice:** The codebase builds on [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Pretrained Stable Diffusion and ControlNet weights use their respective model licenses (e.g. OpenRAIL-M for the ControlNet checkpoint below). Combine obligations accordingly when you redistribute.

---

## Repository layout

| Path | Role |
|------|------|
| `MEG/` | Structural ControlNet + SD img2img: training adapters (`train_controlnet_adapter.py`), batch inference (`batch_infer.py`), visualization helpers |
| `ultralytics/` | Detector fork: RT-DETR yaml, domain head, orthogonality loss, dataloader domain tags |
| `train.py` | CLI training entry (configurable; no hard-coded data paths) |
| `val.py` | Validation / metrics + `paper_data.txt`（CLI：`--weights` / `--data` 必填） |
| `requirements.txt` | Core detector / training deps |
| `requirements-meg.txt` | **MEG-only:** `diffusers`, `transformers`, etc. |
| `notebooks/SRFD-DETR_workflow.ipynb` | Colab-oriented workflow outline |

---

## Quick start (detector only)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Train (you must supply a YOLO-style `data.yaml`):

```bash
python train.py --data path/to/your/data.yaml --cfg ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml --epochs 300 --batch 4 --project runs/train --name my_exp
```

### `train.py` configurable items (CLI)

| Argument | Meaning |
|----------|---------|
| `--data` | **Required.** YOLO `data.yaml` (train/val paths, `nc`, class `names`). |
| `--cfg` | Model yaml (`rtdetr-r18.yaml` = SRFD-style graph; `rtdetr-r18-scale-embed.yaml` = baseline). |
| `--weights` | Optional `.pt` checkpoint to load before training. |
| `--epochs` | Training epochs. |
| `--batch` | Batch size. |
| `--imgsz` | Input image size (default 640). |
| `--workers` | Dataloader workers (use `0` on Windows if unstable). |
| `--cache` | Pass flag to enable RAM cache (`cache=True` in `model.train`). |
| `--device` | e.g. `0` or `0,1`; omit for auto. |
| `--project` | Ultralytics project directory (default `runs/train`). |
| `--name` | Run name under project. |
| `--resume` | Checkpoint path for resume. |

Optional environment variables:

- `RTDETR_DATA` — default `--data` if omitted on CLI (not recommended; prefer explicit `--data`).
- `RTDETR_DEVICE` — e.g. `0` or `0,1`.

### Validation (`val.py`)

```bash
python val.py --weights runs/train/my_exp/weights/best.pt --data path/to/data.yaml --split test --project runs/val --name eval01
```

| Argument | Meaning |
|----------|---------|
| `--weights` | **Required.** `best.pt` 等。可用 `RTDETR_WEIGHTS`。 |
| `--data` | **Required.** `data.yaml`。可用 `RTDETR_DATA`。 |
| `--split` | `train` / `val` / `test`（默认 `test`）。 |
| `--save-json` | 可选标志，写出 COCO JSON。 |
| `--imgsz`, `--batch`, `--project`, `--name`, `--device` | 与训练脚本类似。 |

---

## MEG (diffusion preprocessing)

### Recommended open checkpoints (SD1.5 + Canny ControlNet)

Download once (or let `huggingface_hub` cache on first run):

| Component | Hugging Face repo |
|-----------|-------------------|
| Stable Diffusion 1.5 | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| ControlNet (Canny) | [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny) |

After download, point scripts at **local directories** (offline-friendly; your code often sets `local_files_only=True`):

- `--sd_dir` → folder containing SD1.5 diffusers layout  
- `--controlnet_dir` → folder containing ControlNet weights  

Or pass a merged diffusers pipeline directory to `--pipeline_dir` in `MEG/batch_infer.py` if you export one.

### Install MEG Python deps

```bash
pip install -r requirements-meg.txt
```

### Inference (batch)

Requires **CUDA** (see `MEG/batch_infer.py`). Example:

```bash
python MEG/batch_infer.py --sd_dir path/to/stable-diffusion-v1-5 --controlnet_dir path/to/sd-controlnet-canny --input_dir path/to/sem_images --out_dir path/to/meg_outputs --samples 2 --steps 12
```

### Adapter fine-tuning

Prepare a JSON list (see `MEG/train_controlnet_adapter.py`: `ControlPairDataset`) with `image` and `control` paths, then:

```bash
python MEG/train_controlnet_adapter.py --sd_dir ... --controlnet_dir ... --train_json train.json --output_dir meg_ckpt
```

### Optional: orig vs preprocessed visualization

`MEG/MEG_show.py` — paired folders + YOLO labels (same basename) to export diff figures:

```bash
python MEG/MEG_show.py --orig_dir ./images/train --prep_dir ./preproc/train --labels_dir ./labels/train --out_prefix ./viz/run1 --max_samples 20
```

### Pack SD + ControlNet for offline `batch_infer`

```bash
python MEG/build_start_pipeline.py --sd_dir path/to/stable-diffusion-v1-5 --controlnet_dir path/to/sd-controlnet-canny --out path/to/sd15_canny_img2img_pipeline
```

Then pass `--pipeline_dir path/to/sd15_canny_img2img_pipeline` to `MEG/batch_infer.py` if you prefer a single merged folder.

---

## Model configs vs. paper / ablations

| YAML | Purpose |
|------|---------|
| `ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml` | **SRFD-style** dual branch at P5 + domain classifier + orthogonality loss in graph |

Detection loss weights for domain / disentanglement are wired in code (`ultralytics/models/utils/loss.py`, `lambda_disentangle` in yaml where applicable).

---

## Dataset and domain labels

- Standard Ultralytics/YOLO detection layout: images + labels, `data.yaml` with `train`, `val`, `names`, `nc`, etc.
- **Domain supervision:** `ultralytics/data/dataset.py` assigns `domain` from image file path (`nm` → 0, `μm` / `um` style patterns → 1, else -1). Adjust naming conventions or code if your folder scheme differs.

---

## Colab / Jupyter workflow

See **`notebooks/SRFD-DETR_workflow.ipynb`** for a step-by-step outline: install → (optional) clone repo → Hugging Face auth → MEG inference → detector train/val commands.

