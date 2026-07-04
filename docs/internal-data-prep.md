# 内部数据整理说明

当医院发来的图片命名混乱、层级混乱或夹杂无关文件时，先用整理脚本生成一个干净的标注文件夹，再发给医院使用。

## 使用方式

```powershell
python scripts\prepare_hospital_dataset.py `
  --input "D:\hospital_raw" `
  --output "D:\hip22_clean_for_annotation" `
  --prefix "case"
```

输出目录会包含：

- `case_0001.jpg` 这类统一命名后的图片。
- `rename_map.csv`：新文件名和原始路径的对应关系。
- `issues.csv`：无法读取或需要人工处理的文件。
- 如果原图旁边已有同名 `.txt` 标注，会复制为新图片同名 `.txt`。

## 交付给医院

把输出目录整体发给医院，让医生直接导入该文件夹标注。

如果 `issues.csv` 中有内容，先人工确认这些文件是否需要补传、转换格式或排除。
