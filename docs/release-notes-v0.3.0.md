# Hip22AnnotationTool v0.3.0

Label: non-production ready

This release adds manual completion confirmation for both keypoints and Shenton lines, prevents
auto-filling missing model points with template fallback data, and improves the review panel layout.

## Highlights

- **Manual keypoint confirmation.** Reviewers must explicitly click "确认关键点完成" before
  keypoints count as complete. All 22 model-detected visible points are treated as review-ready
  (status `auto`), not complete.
- **Manual Shenton confirmation.** A matching "确认沈通线完成" button requires explicit reviewer
  sign-off for Shenton curves, consistent with the keypoint confirmation workflow.
- **Template fallback disabled.** When model output is unavailable or incomplete, missing landmarks
  stay explicitly missing (`visible=False`, `source="missing"`) instead of being filled with
  draggable template guesses. Reviewers must manually place only verified points.
- **Collapsible Shenton panel.** The Shenton & Measurement section is collapsed by default, styled
  consistently with the Advanced Settings panel.
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
