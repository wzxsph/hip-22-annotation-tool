# Hip22AnnotationTool v0.3.0

Label: non-production ready

This release adds manual completion confirmation for keypoints and Shenton lines, 14 default
anatomical guide connections, and disables template fallback for missing model points.

## Highlights

- **Manual keypoint confirmation.** Reviewers must explicitly click "确认关键点完成" before
  keypoints count as complete. All 22 model-detected visible points are treated as review-ready
  (status `auto`), not complete.
- **Manual Shenton confirmation.** A matching "确认沈通线完成" button requires explicit reviewer
  sign-off for Shenton curves, consistent with the keypoint confirmation workflow. Shenton &
  Measurement panel is collapsed by default with the same styling as Advanced Settings.
- **14 default guide connections.** Predefined anatomical connections (acetabular, femoral head,
  femoral shaft, cross-midline) are drawn in light red on the canvas. Enabled by default with
  a toggle under Display settings. Manual connections remain independently toggleable in yellow.
- **Template fallback disabled.** When model output is unavailable or incomplete, missing landmarks
  stay explicitly missing (`visible=False`, `source="missing"`) instead of being filled with
  draggable template guesses. Reviewers must manually place only verified points.
- **New `shenton_awaiting_confirmation` status.** Shenton curves that are fully drawn and reviewed
  on both sides but not yet confirmed by the reviewer show as a distinct filter state.

## Status Workflow (v0.3.0)

| Status | Meaning |
|--------|---------|
| `pending` | No keypoints visible, no Shenton started |
| `auto` | All 22 keypoints model-detected, no manual edits, no Shenton started |
| `in_progress` | Some keypoints placed or Shenton started, not yet confirmed |
| `keypoint_complete` | Keypoints manually confirmed |
| `shenton_awaiting_confirmation` | Both Shenton sides complete with doctor review, not yet confirmed |
| `shenton_complete` | Shenton manually confirmed |
| `done` | Both keypoints and Shenton confirmed |

## Default Guide Connections

| Pair | Description |
|------|-------------|
| 左5 - 左6 | 股骨干轴近端-远端 |
| 左8 - 左3 - 左9 | 股骨头内侧-中心-外侧 |
| 左3 - 左7 | 股骨头中心-股骨颈轴中心 |
| 左1 - 左2 | 髋臼外上缘-Y软骨中心 |
| 左1 - 左4 | 髋臼外上缘-泪滴下缘 |
| (same for right) | |
| 左2 - 右2 | Y软骨中心（跨中线） |
| 左4 - 右4 | 泪滴下缘（跨中线） |

## Breaking Changes from v0.2.0

- Annotations saved by v0.2.0 that had `source="template_guess"` keypoints are still loadable
  (legacy compatibility), but new auto-detection output no longer creates template guesses.
- `review.manual_keypoints_complete` and `review.manual_shenton_complete` fields are now required
  for `complete` / `done` status. Annotations without these fields default to not-complete.
- The `auto_initialization.template_fallback` field now writes `enabled: false` with a
  compatibility note instead of filling points.

## Verification

Passed locally:

```text
uv run pytest
uv run python -m compileall annotation_tool
node --check static/app.js
```
