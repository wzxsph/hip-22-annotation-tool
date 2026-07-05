import numpy as np
from PIL import Image

from annotation_tool.heuristics import AutoAnnotationResult
from annotation_tool.scan_like import map_result_from_scan, warp_scan_like_image
from annotation_tool.schema import create_blank_annotation, make_keypoint


def test_warp_scan_like_image_maps_points_back_to_original_coordinates():
    image = Image.new("RGB", (120, 90), color="black")
    corners = np.asarray([(20, 10), (100, 15), (95, 75), (25, 80)], dtype=np.float32)

    warp = warp_scan_like_image(image, corners)

    assert warp.image.width > 60
    assert warp.image.height > 55
    detected = create_blank_annotation("case.png", warp.image.width, warp.image.height)
    detected.keypoints["left_acetabular_outer"] = make_keypoint(
        "left",
        "acetabular_outer",
        0,
        0,
        source="pose11_side",
        confidence=0.9,
    )
    result = AutoAnnotationResult(
        keypoints=detected.keypoints,
        warnings=[],
        model_available=True,
        source="yolo11n-best-side11",
    )

    mapped = map_result_from_scan(result, warp.inverse_matrix)
    point = mapped.keypoints["left_acetabular_outer"]

    assert point.x == 20
    assert point.y == 10
