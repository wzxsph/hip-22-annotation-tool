from annotation_tool.completion import annotation_progress
from annotation_tool.schema import create_blank_annotation


def test_manual_keypoint_completion_allows_missing_points():
    annotation = create_blank_annotation("case.png", 100, 100)
    annotation.review["manual_keypoints_complete"] = {
        "status": "confirmed",
        "updated_at": "2026-07-06T00:00:00Z",
        "annotator": "doctor-a",
        "note": "Confirmed despite missing points.",
    }

    progress = annotation_progress(annotation)

    assert progress["keypoint_status"] == "complete"
    assert progress["keypoints"]["visible"] == 0
    assert progress["keypoints"]["manual_confirmed"] is True


def test_manual_shenton_completion_allows_incomplete_curves():
    annotation = create_blank_annotation("case.png", 100, 100)
    annotation.review["manual_shenton_complete"] = {
        "status": "confirmed",
        "updated_at": "2026-07-06T00:00:00Z",
        "annotator": "doctor-a",
        "note": "Confirmed because Shenton is not assessable.",
    }

    progress = annotation_progress(annotation)

    assert progress["shenton_status"] == "complete"
    assert progress["shenton"]["complete_sides"] == 0
    assert progress["shenton"]["manual_confirmed"] is True
