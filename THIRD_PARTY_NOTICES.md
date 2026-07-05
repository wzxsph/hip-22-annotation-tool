# Third-Party Notices

## Ultralytics YOLO

This repository bundles a pose weight derived from the Ultralytics YOLO model family and uses the `ultralytics` Python package for inference.

Ultralytics provides an AGPL-3.0 path and a separate Enterprise License path. This public repository is distributed under AGPL-3.0. For closed-source or commercial embedded use, review the Ultralytics license terms and obtain the appropriate license if needed.

- License page: https://www.ultralytics.com/license
- Python package: https://pypi.org/project/ultralytics/

## MTDDH Dataset

Demo screenshots use pelvic X-ray examples derived from the MTDDH dataset, licensed under CC BY 4.0; see Qi et al., Scientific Data, 2025.

This repository does not redistribute the original MTDDH image dataset. Screenshots are included only to demonstrate the annotation interface.

- Article: https://www.nature.com/articles/s41597-025-05146-x
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
- Data record: https://data.niaid.nih.gov/resources?id=figshare_27999131

Suggested citation:

> Qi et al. MTDDH: a multi-task developmental dysplasia of the hip dataset for pelvic X-ray analysis. Scientific Data, 2025.

## FastAPI and Python Runtime Packages

The annotation server and UI use common Python and browser packages including FastAPI, Uvicorn, Pillow, NumPy, OpenCV, PyTest, pydicom, and browser-native Canvas APIs. See `requirements.txt` and `pyproject.toml` for the install list.
