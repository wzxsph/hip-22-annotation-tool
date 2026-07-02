# Hip 22-Point Annotation Tool

FastAPI + Canvas web tool for reviewing and editing 22 hip X-ray landmarks, with optional model-assisted initialization.

This project is a research annotation aid. It is not a medical device, does not provide diagnosis, and should not be used as a standalone clinical decision system. All automatically generated points should be reviewed and corrected by a qualified user before downstream use.

![Annotation workspace](docs/screenshots/workspace-overview.png)

## What It Does

- Annotates 22 landmarks: image-left and image-right, each with hospital points #1-#11.
- Loads a local image folder as a workspace.
- Preserves existing JSON or YOLO sidecar labels and never overwrites reviewed annotations during auto-detection.
- Uses the bundled `models/yolo11n-best.pt` weight by default for model-assisted initialization.
- Saves complete annotation JSON to `annotations/<image_stem>.json`.
- Saves YOLO Pose sidecar labels to `<image_stem>.txt` beside each image.
- Supports zoom, pan, drag-to-correct, missing-point marking, undo/redo, manual connections, and keyboard image navigation.

The bundled yolo11n-best weight is trained from manually created research annotations on MTDDH-derived images by a non-medical annotator; users should review and correct all outputs before use.

## Demo Screenshots

Demo screenshots use pelvic X-ray examples derived from the MTDDH dataset, licensed under CC BY 4.0; see Qi et al., Scientific Data, 2025. The original MTDDH image dataset is not redistributed in this repository.

| Workspace | Point Review | Export |
|---|---|---|
| ![Workspace overview](docs/screenshots/workspace-overview.png) | ![Point editing](docs/screenshots/point-editing.png) | ![YOLO export](docs/screenshots/export-panel.png) |

## Install

Python 3.10-3.12 is recommended. The bundled model uses the `ultralytics` package for inference.

```bash
git clone https://github.com/<your-name>/hip-22-annotation-tool.git
cd hip-22-annotation-tool

uv venv --python python3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv run pytest
```

Start the local server:

```bash
uv run uvicorn annotation_tool.server:app --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010/`.

## Model Setup

Default model path:

```text
models/yolo11n-best.pt
```

You can override it:

```bash
HIP22_MODEL_PATH=/absolute/path/to/model.pt \
uv run uvicorn annotation_tool.server:app --host 127.0.0.1 --port 8010
```

Device selection:

- `HIP22_DEVICE=auto` or unset: use `cuda:0` when PyTorch reports CUDA is available; otherwise use CPU.
- `HIP22_DEVICE=cpu`: force CPU inference.
- `HIP22_DEVICE=cuda:0`: force a specific GPU.

If the model file is missing or `ultralytics` is unavailable, the tool still opens images and creates a blank 22-point template with a visible warning instead of returning an internal server error.

## Workspace Layout

Importing a folder makes that folder the current workspace. The tool may create these files in that folder:

```text
workspace/
├── data.yaml
├── manifest.json
├── <image_stem>.txt
├── annotations/
│   └── <image_stem>.json
└── splits/
    └── train_val_split.json
```

Existing data priority:

1. `annotations/<image_stem>.json`
2. same-folder `<image_stem>.txt`
3. blank template or model-assisted initialization

Existing JSON and imported sidecar labels are treated as user data and are not overwritten by auto-detection.

## Landmark Schema

Each image stores 22 keypoints. `left_*` means image-left and `right_*` means image-right; there is no anatomical left/right swap.

| # | Field | Chinese label |
|---|---|---|
| 1 | `acetabular_outer` | 髋臼外上缘 |
| 2 | `triradiate_center` | Y 形软骨中心 |
| 3 | `femoral_head_center` | 股骨头中心 |
| 4 | `teardrop_lower` | 泪滴下缘 |
| 5 | `femoral_shaft_prox` | 股骨干轴近端中心 |
| 6 | `femoral_shaft_dist` | 股骨干轴远端中心 |
| 7 | `femoral_neck_axis_center` | 股骨颈轴中心 |
| 8 | `femoral_head_medial` | 股骨头最内侧缘 |
| 9 | `femoral_head_lateral` | 股骨头最外侧缘 |
| 10 | `obturator_upper` | 闭孔上缘 |
| 11 | `femoral_neck_inner_lower` | 股骨颈内下缘 |

Missing points are represented explicitly:

```json
{
  "x": null,
  "y": null,
  "visible": false,
  "visibility": 0,
  "source": "missing"
}
```

The UI treats a point as visible only when `visible = true`, `visibility > 0`, and both `x` and `y` are present. Source strings do not determine missingness.

## Controls

| Action | Control |
|---|---|
| Drag point | Move and mark as manual |
| Right-click canvas | Quickly place selected point |
| Mouse wheel | Zoom |
| Space + drag | Pan |
| `←` / `→` | Previous / next image |
| `Delete` | Mark selected point missing, or hide/delete selected connection |
| `Ctrl+S` | Save |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |
| `P` / `V` / `L` | Point / select / line mode |
| `H` | Toggle point labels |

## YOLO Pose Label Format

The tool saves one same-folder `.txt` per image. Each file has 11 rows:

```text
class_id cx cy w h left_x left_y left_vis right_x right_y right_vis
```

Rules:

- `class_id` is #1-#11 in zero-based order.
- Coordinates are normalized to `[0, 1]`.
- Visible points use `vis=2`.
- Missing points use `0 0 0`.
- `data.yaml` uses `kpt_shape: [2, 3]` and `flip_idx: [1, 0]`.

## License and Attribution

This repository is distributed under GNU AGPL-3.0. The bundled model weight is derived from the Ultralytics YOLO model family and is distributed with this repository under AGPL-3.0. For closed-source or commercial embedded use, review the Ultralytics license terms and obtain the appropriate license if needed.

MTDDH attribution:

> Demo screenshots use pelvic X-ray examples derived from the MTDDH dataset, licensed under CC BY 4.0; see Qi et al., Scientific Data, 2025.

Links:

- Ultralytics license: https://www.ultralytics.com/license
- MTDDH article: https://www.nature.com/articles/s41597-025-05146-x
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
- Model details: [MODEL_CARD.md](MODEL_CARD.md)
- Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Development Checks

```bash
uv run pytest
uv run python -m compileall annotation_tool
```

Before publishing a public repository, check that runtime data is absent:

```bash
find . -maxdepth 3 \( -name '.venv' -o -name 'manifest.json' -o -name 'tool-settings.json' -o -name 'annotations' -o -name 'images' -o -name 'labels' \) -print
git lfs ls-files
```
