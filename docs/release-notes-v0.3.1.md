# Hip22AnnotationTool v0.3.1

Label: non-production ready

This patch release streamlines the default review view and makes image deletion recoverable.

## Changes

- Default guide connections now exclude both-side #1-#4 lines and the left #4-right #4 cross-midline line.
- #10 and #11 points are hidden on the canvas by default to reduce visual clutter. They remain in the 22-point schema and can be shown from Advanced Settings.
- "Delete image" now moves the image plus same-name annotation JSON and YOLO txt files into a `trash` folder beside the image. Manifest entries are removed from the active list, and cache previews are still cleared.
- 16-bit TIFF display and enhanced preview are normalized from source pixels so X-ray contrast enhancement no longer collapses high-bit-depth TIFFs to white.

## Default Guide Connections

| Pair | Description |
|------|-------------|
| 左5 - 左6 | 股骨干轴近端-远端 |
| 左8 - 左3 - 左9 | 股骨头内侧-中心-外侧 |
| 左3 - 左7 | 股骨头中心-股骨颈轴中心 |
| 左1 - 左2 | 髋臼外上缘-Y软骨中心 |
| (same for right) | |
| 左2 - 右2 | Y软骨中心（跨中线） |

## Recovery Note

Files moved by "删除本图" are placed under `trash` in the same folder as the image. To restore manually, move the image back to its original folder and move the associated JSON/txt files back to their recorded source locations.

## Verification

Passed locally:

```text
uv run pytest
uv run python -m compileall annotation_tool
node --check static/app.js
```
