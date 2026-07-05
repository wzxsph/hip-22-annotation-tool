import json

from PIL import Image

from scripts.prepare_scan_like_images import prepare_scan_like_dataset


def test_prepare_scan_like_dataset_writes_clean_images_and_mapping(tmp_path):
    input_root = tmp_path / "raw"
    output_root = tmp_path / "out"
    input_root.mkdir()
    Image.new("RGB", (120, 90), color="black").save(input_root / "phone case.png")
    corners = {
        "phone case.png": [
            {"x": 20, "y": 10},
            {"x": 100, "y": 15},
            {"x": 95, "y": 75},
            {"x": 25, "y": 80},
        ]
    }
    corners_path = tmp_path / "corners.json"
    corners_path.write_text(json.dumps(corners), encoding="utf-8")

    report = prepare_scan_like_dataset(input_root, output_root, corners_json=corners_path)

    assert report["written"] == 1
    assert report["failed"] == 0
    assert (output_root / "scan_like_mapping.csv").exists()
    assert (output_root / "scan_like_report.json").exists()
    output_images = list(output_root.glob("scanlike_*.png"))
    assert len(output_images) == 1
    assert report["records"][0]["mode"] == "manual_corners"
    assert report["records"][0]["scan_transform"]["enabled"] is True
