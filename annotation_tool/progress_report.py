from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
from typing import Any

from .schema import Annotation, Manifest, ManifestImage, annotation_from_dict


STATUS_LABELS = {
    "pending": "未标注",
    "auto": "自动初标待复核",
    "in_progress": "人工修改中",
    "done": "已完成",
}
STATUS_ORDER = ("pending", "auto", "in_progress", "done")
STATUS_MARKER_GLOB = "HIP22_STATUS_DONE_*_TODO_*.txt"
HTML_REPORT_NAME = "HIP22_status_report.html"
CSV_REPORT_NAME = "HIP22_status_report.csv"
SUBMISSION_README_NAME = "HIP22_SUBMISSION_README.txt"


def build_progress_payload(root: Path, manifest: Manifest | None = None) -> dict[str, Any]:
    rows = build_progress_rows(root, manifest)
    return _payload_from_counts(progress_counts(rows))


def write_progress_reports(root: Path, manifest: Manifest | None = None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    rows = build_progress_rows(root, manifest)
    counts = progress_counts(rows)
    payload = _payload_from_counts(counts)

    root.mkdir(parents=True, exist_ok=True)
    for old_marker in root.glob(STATUS_MARKER_GLOB):
        if old_marker.is_file():
            old_marker.unlink()

    _write_text(root / payload["report_files"]["marker"], _status_marker_text(counts))
    _write_text(root / HTML_REPORT_NAME, _html_report(rows, counts))
    _write_text(root / CSV_REPORT_NAME, _csv_report(rows))
    _write_text(root / SUBMISSION_README_NAME, _submission_readme_text(counts))
    return payload


def build_progress_rows(root: Path, manifest: Manifest | None = None) -> list[dict[str, Any]]:
    if manifest is None:
        from .storage import load_manifest

        manifest = load_manifest(root)

    rows = [_row_for_manifest_item(root, item) for item in manifest.images]
    rows.sort(key=lambda row: (STATUS_ORDER.index(row["status"]) if row["status"] in STATUS_ORDER else 99, row["filename"]))
    return rows


def progress_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUS_ORDER}
    for row in rows:
        status = row.get("status") if row.get("status") in counts else "pending"
        counts[status] += 1
    counts["total"] = len(rows)
    counts["needs_review"] = counts["pending"] + counts["auto"] + counts["in_progress"]
    return counts


def _payload_from_counts(counts: dict[str, int]) -> dict[str, Any]:
    done = counts["done"]
    todo = counts["total"] - done
    return {
        "counts": counts,
        "labels": STATUS_LABELS,
        "report_files": {
            "marker": f"HIP22_STATUS_DONE_{done}_TODO_{todo}.txt",
            "html": HTML_REPORT_NAME,
            "csv": CSV_REPORT_NAME,
            "readme": SUBMISSION_README_NAME,
        },
    }


def _row_for_manifest_item(root: Path, item: ManifestImage) -> dict[str, Any]:
    filename = Path(item.image_path).name
    annotation_path = _annotation_path(root, item)
    label_path = _label_path(root, item)
    annotation = _load_annotation_safely(annotation_path)
    warnings: list[str] = []
    auto_source = ""
    visible_count = 0
    manual_count = 0

    if annotation is not None:
        visible_count, manual_count = _annotation_counts(annotation)
        status = _status_from_counts(visible_count, manual_count)
        auto_source = str(annotation.auto_initialization.get("source") or "")
        warnings = [str(item) for item in annotation.auto_initialization.get("warnings", [])]
    else:
        status = item.status if item.status in STATUS_LABELS else "pending"
        if annotation_path.exists():
            warnings.append("标注 JSON 无法读取，请联系整理人员。")
        elif status != "pending":
            status = "pending"

    return {
        "filename": filename,
        "status": status,
        "status_label": STATUS_LABELS.get(status, STATUS_LABELS["pending"]),
        "visible_count": visible_count,
        "manual_count": manual_count,
        "auto_source": auto_source,
        "warnings": "; ".join(warnings),
        "json_exists": annotation_path.exists(),
        "txt_exists": label_path.exists(),
        "annotation_path": _relative_text(annotation_path, root),
        "label_path": _relative_text(label_path, root),
    }


def _annotation_counts(annotation: Annotation) -> tuple[int, int]:
    visible_count = 0
    manual_count = 0
    for point in annotation.keypoints.values():
        visible = bool(point.visible and point.x is not None and point.y is not None)
        if not visible:
            continue
        visible_count += 1
        if point.source == "manual":
            manual_count += 1
    return visible_count, manual_count


def _status_from_counts(visible_count: int, manual_count: int) -> str:
    if visible_count == 0:
        return "pending"
    if manual_count == 0:
        return "auto"
    if visible_count == 22:
        return "done"
    return "in_progress"


def _annotation_path(root: Path, item: ManifestImage) -> Path:
    configured = root / item.annotation_path
    if configured.exists():
        return configured
    return root / "annotations" / f"{Path(item.image_path).stem}.json"


def _label_path(root: Path, item: ManifestImage) -> Path:
    image_path = root / item.image_path
    stem = Path(item.image_path).stem
    candidates = [
        image_path.with_suffix(".txt"),
        root / f"{stem}.txt",
        root / "labels" / "train" / f"{stem}.txt",
        root / "labels" / "val" / f"{stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_annotation_safely(path: Path) -> Annotation | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return annotation_from_dict(json.load(handle))
    except Exception:
        return None


def _relative_text(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _status_marker_text(counts: dict[str, int]) -> str:
    return (
        "Hip22 标注进度\n"
        f"总数：{counts['total']} 张\n"
        f"已完成：{counts['done']} 张\n"
        f"待处理：{counts['needs_review']} 张\n"
        f"未标注：{counts['pending']} 张\n"
        f"自动初标待复核：{counts['auto']} 张\n"
        f"人工修改中：{counts['in_progress']} 张\n\n"
        "请打开 HIP22_status_report.html 查看每张图片的状态。\n"
        "医生标注完成后，保存并将整个文件夹发送给项目团队即可。\n"
    )


def _submission_readme_text(counts: dict[str, int]) -> str:
    return (
        "Hip22 医院标注文件夹说明\n\n"
        "请按以下流程使用：\n"
        "1. 将需要标注的图片统一放在这个文件夹中。\n"
        "2. 在软件中导入这个文件夹，等待自动识别完成。\n"
        "3. 逐张打开图片，拖拽或补充关键点。\n"
        "4. 点击保存，确认进度报告更新。\n"
        "5. 标注结束后，将整个文件夹直接发送给项目团队。\n\n"
        f"当前总数：{counts['total']} 张\n"
        f"已完成：{counts['done']} 张\n"
        f"仍需处理：{counts['needs_review']} 张\n\n"
        "如果文件命名混乱、目录嵌套复杂、无法导入，或不确定如何整理，可先将照片发给项目团队整理后再标注。\n"
    )


def _csv_report(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "filename",
            "status_label",
            "visible_count",
            "manual_count",
            "auto_source",
            "warnings",
            "json_exists",
            "txt_exists",
            "annotation_path",
            "label_path",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    return output.getvalue()


def _html_report(rows: list[dict[str, Any]], counts: dict[str, int]) -> str:
    summary_cards = "".join(
        f"<div><span>{html.escape(STATUS_LABELS[status])}</span><strong>{counts[status]}</strong></div>"
        for status in STATUS_ORDER
    )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['filename'])}</td>"
        f"<td><span class='badge {html.escape(row['status'])}'>{html.escape(row['status_label'])}</span></td>"
        f"<td>{row['visible_count']}/22</td>"
        f"<td>{row['manual_count']}</td>"
        f"<td>{html.escape(row['auto_source'] or '-')}</td>"
        f"<td>{html.escape(row['warnings'] or '-')}</td>"
        f"<td>{'是' if row['json_exists'] else '否'}</td>"
        f"<td>{'是' if row['txt_exists'] else '否'}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>Hip22 标注进度</title>
    <style>
      body {{ font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; margin: 24px; color: #182026; }}
      h1 {{ margin: 0 0 8px; font-size: 24px; }}
      p {{ color: #5d6972; }}
      .summary {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin: 20px 0; }}
      .summary div {{ border: 1px solid #d7dee2; border-radius: 8px; padding: 12px; background: #f7f9fa; }}
      .summary span {{ display: block; color: #68747d; font-size: 13px; }}
      .summary strong {{ display: block; margin-top: 6px; font-size: 26px; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
      th, td {{ border: 1px solid #d7dee2; padding: 8px; text-align: left; vertical-align: top; }}
      th {{ background: #eef2f3; }}
      .badge {{ display: inline-block; border-radius: 999px; padding: 3px 8px; color: white; }}
      .pending {{ background: #868e96; }}
      .auto {{ background: #6741d9; }}
      .in_progress {{ background: #e67700; }}
      .done {{ background: #2f9e44; }}
    </style>
  </head>
  <body>
    <h1>Hip22 标注进度</h1>
    <p>保存后将整个文件夹发送给项目团队即可。未标注、自动初标待复核、人工修改中都属于仍需处理。</p>
    <section class="summary">{summary_cards}</section>
    <table>
      <thead>
        <tr>
          <th>文件名</th>
          <th>状态</th>
          <th>可见点</th>
          <th>人工点</th>
          <th>自动来源</th>
          <th>提示</th>
          <th>JSON</th>
          <th>TXT</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </body>
</html>
"""
