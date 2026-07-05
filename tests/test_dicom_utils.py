from pathlib import Path

from annotation_tool.dicom_utils import is_dicom_path, read_dicom_image

from conftest import write_test_dicom


def test_read_dicom_image_extracts_spacing_and_renders_rgb(tmp_path):
    path = tmp_path / "case.dcm"
    write_test_dicom(path)

    result = read_dicom_image(path)

    assert result.image.mode == "RGB"
    assert result.image.size == (16, 12)
    assert result.metadata["source_format"] == "dicom"
    assert result.metadata["pixel_spacing_row_mm"] == 0.2
    assert result.metadata["pixel_spacing_col_mm"] == 0.3
    assert result.metadata["pixel_spacing_source"] == "PixelSpacing"


def test_dicom_magic_allows_extensionless_files(tmp_path):
    path = tmp_path / "case_without_extension"
    write_test_dicom(path)

    assert is_dicom_path(path) is True
    assert read_dicom_image(path).image.size == (16, 12)


def test_monochrome1_dicom_is_inverted_for_display(tmp_path):
    mono2 = tmp_path / "mono2.dcm"
    mono1 = tmp_path / "mono1.dcm"
    write_test_dicom(mono2, monochrome1=False)
    write_test_dicom(mono1, monochrome1=True)

    left_mono2 = read_dicom_image(mono2).image.getpixel((0, 0))[0]
    left_mono1 = read_dicom_image(mono1).image.getpixel((0, 0))[0]

    assert left_mono1 > left_mono2
