# Hip22AnnotationTool v0.1.0 Windows CPU 医院交付版

## 下载与安装

医院端请下载 Release 中的 `Hip22AnnotationTool-v0.1.0-win64-cpu.zip`，解压后运行 `Hip22AnnotationTool.exe` 或 `Run-Hip22.bat`。该包按普通 Windows 10/11、无管理员权限、离线使用场景准备，默认使用 CPU 推理。

不要把 GitHub 页面里的 `Code > Download ZIP` 当作医院交付包使用。源码 ZIP 不可靠地包含 Git LFS 大模型文件，可能导致 `models/yolo11n-best.pt` 缺失或变成一个很小的 LFS 指针文件，从而出现自动识别不可用。开发者需要使用 `git clone` 并执行 `git lfs pull`。

## 医生主流程

1. 将需要标注的图片统一放到一个文件夹中。
2. 打开工具并导入整个文件夹，系统会自动识别其中支持的图片。
3. 等待自动初标；如果某张图未识别，可以点击当前图片的“自动识别”按钮重试。
4. 拖拽标注点进行复核和修改。
5. 点击保存，确认 JSON/TXT 已写入。
6. 将整个标注完成的文件夹发回。

如果图片命名混乱、目录嵌套混乱、无法导入或不确定怎么整理，可以先把照片发给项目方整理，整理后再返给医院标注。

## 本版变化

- 重建 Windows CPU ZIP 交付流程，保留本地 FastAPI + 浏览器 Canvas UI。
- 修复 PyInstaller 打包时模型依赖缺失导致的 `model-unavailable` 风险。
- 增加标注进度总览、状态筛选和文件夹内状态报告。
- 增加自动识别多次降级尝试，失败时保留未标注状态并提示需要手工标注。
- 增加当前图片“自动识别”按钮，便于医生对未成功识别的图片手动重试，且不会覆盖人工修改点。
- 强化医生端提交提示：保存后直接发送整个标注文件夹。
- 增加内部数据整理脚本和中文使用文档、演示视频脚本。

## 验收记录

- `python -m pytest`：27 passed。
- `python -m compileall annotation_tool`：通过。
- `node --check static/app.js`：通过。
- Windows ZIP smoke test：`/api/health`、首页、静态 JS、schema 和启动流程通过。

## 隐私与数据

本 Release 不包含医院真实数据、患者隐私图片或原始训练数据。仓库中的演示截图可以上传；本地 X 光片、MTDDH 原始图片和医院数据文件夹不得提交到 git。
