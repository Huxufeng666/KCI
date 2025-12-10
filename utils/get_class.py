# # generate_labels_csv.py
# import os
# import pandas as pd
# from pathlib import Path

# # ==================== 配置区域 ====================
# ROOT_DIR = "/workspace/dataset"          # 你的数据集根目录
# OUTPUT_CSV = "labels.csv"     # 输出文件名
# SPLITS = ['train', 'val', 'test']  # 要处理的子集
# # ===============================================

# CLASS_MAP = {
#     'benign':    {'name': 'Benign',    'id': 0},
#     'malignant': {'name': 'Malignant', 'id': 1},
#     'normal':    {'name': 'Normal',    'id': 2}
# }

# def parse_filename(fname):
#     """从 benign (1).png 提取 class 和 id"""
#     fname = fname.lower()
#     for cls_key in CLASS_MAP.keys():
#         if cls_key in fname:
#             try:
#                 start = fname.index('(') + 1
#                 end = fname.index(')')
#                 num_str = fname[start:end].strip()
#                 num = int(num_str)
#                 return cls_key, num
#             except:
#                 continue
#     return None, None

# def main():
#     records = []

#     for split in SPLITS:
#         img_dir = Path(ROOT_DIR) / split / 'images'
#         mask_dir = Path(ROOT_DIR) / split / 'masks'

#         if not img_dir.exists():
#             print(f"跳过 {split}: {img_dir} 不存在")
#             continue

#         print(f"正在处理 {split}...")

#         for img_path in img_dir.rglob("*.png"):
#             if img_path.name.endswith('_mask.png'):
#                 continue

#             cls_key, num = parse_filename(img_path.stem)
#             if cls_key is None:
#                 print(f"警告: 无法解析类别: {img_path}")
#                 continue

#             class_info = CLASS_MAP[cls_key]
#             mask_name = f"{img_path.stem}_mask.png"
#             mask_path = mask_dir / mask_name

#             if not mask_path.exists():
#                 print(f"警告: 找不到 mask: {mask_path}")
#                 continue

#             # ========== 修复：必须加 class_name ==========
#             records.append({
#                 'image_path': str(img_path),
#                 'mask_path': str(mask_path),
#                 'class_name': class_info['name'],   # ← 必须加！
#                 'class_id': class_info['id']
#             })

#     if not records:
#         print("错误：没有找到任何样本！请检查路径和文件名格式")
#         return

#     # ========== 修复：确保列存在 ==========
#     df = pd.DataFrame(records)
#     print(f"原始列: {df.columns.tolist()}")  # 调试用

#     # 安全排序
#     if 'class_name' in df.columns:
#         df = df.sort_values(['class_name', 'image_path']).reset_index(drop=True)
#     else:
#         print("错误：DataFrame 中没有 'class_name' 列！")
#         print(df.head())
#         return

#     df.to_csv(OUTPUT_CSV, index=False)
#     print(f"\n成功生成 {OUTPUT_CSV}，共 {len(df)} 条记录")
#     print(f"类别分布：\n{df['class_name'].value_counts()}")

#     # ========== 额外：检查 normal 肿瘤 ==========
#     print("\n检查 normal 类是否有肿瘤...")
#     import cv2
#     errors = []
#     for _, row in df.iterrows():
#         if row['class_name'] == 'Normal':
#             mask = cv2.imread(row['mask_path'], 0)
#             if mask is not None:
#                 area = (mask > 127).sum()
#                 if area > 100:
#                     errors.append((row['image_path'], area))

#     if errors:
#         print(f"发现 {len(errors)} 个 normal 类有肿瘤！建议移到 benign/")
#         for path, area in errors[:5]:
#             print(f"  {path} → 面积 {area}")
#     else:
#         print("Normal 类全部无肿瘤，完美！")

# if __name__ == "__main__":
#     main()




# generate_labels_csv.py
import os
import cv2
import pandas as pd
from pathlib import Path

# ==================== 配置 ====================
ROOT_DIR = "/workspace/dataset"
OUTPUT_CSV = "labels_fixed.csv"
SPLITS = ['train', 'val', 'test']
# ===============================================

def infer_class_from_mask(mask_path, area_threshold=100):
    """根据 mask 推断真实类别"""
    if not mask_path.exists():
        return 'Normal', 2

    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        return 'Normal', 2

    area = (mask > 127).sum()
    if area == 0:
        return 'Normal', 2
    elif area < area_threshold:
        return 'Benign', 0
    else:
        return 'Malignant', 1  # 大肿瘤 → 恶性

def main():
    records = []

    for split in SPLITS:
        img_dir = Path(ROOT_DIR) / split / 'images'
        mask_dir = Path(ROOT_DIR) / split / 'masks'

        if not img_dir.exists():
            print(f"跳过 {split}")
            continue

        print(f"处理 {split}...")

        for img_path in img_dir.rglob("*.png"):
            if '_mask' in img_path.name:
                continue

            # 构造 mask 路径
            mask_name = f"{img_path.stem}_mask.png"
            mask_path = mask_dir / mask_name

            # 推断真实类别（忽略文件名）
            true_class_name, true_class_id = infer_class_from_mask(mask_path)

            # 解析文件名类别（用于对比）
            fname_lower = img_path.stem.lower()
            file_class = 'unknown'
            if 'benign' in fname_lower:
                file_class = 'Benign'
            elif 'malignant' in fname_lower:
                file_class = 'Malignant'
            elif 'normal' in fname_lower:
                file_class = 'Normal'

            records.append({
                'image_path': str(img_path),
                'mask_path': str(mask_path) if mask_path.exists() else '',
                'file_class': file_class,           # 文件名标的
                'true_class': true_class_name,      # mask 推断的
                'true_class_id': true_class_id,
                'mask_area': (cv2.imread(str(mask_path), 0) > 127).sum() if mask_path.exists() else 0
            })

    df = pd.DataFrame(records)

    # 标记冲突
    df['conflict'] = df['file_class'] != df['true_class']
    conflicts = df[df['conflict']]
    print(f"\n发现 {len(conflicts)} 个标签冲突！")

    # 保存完整报告
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"已保存: {OUTPUT_CSV}")

    # 打印冲突样本
    if len(conflicts) > 0:
        print("\n冲突样本（前10个）：")
        for _, row in conflicts.head(10).iterrows():
            print(f"  文件: {row['image_path']}")
            print(f"    文件名标: {row['file_class']} | 真实: {row['true_class']} (面积={row['mask_area']})")

    # 统计
    print(f"\n真实类别分布：\n{df['true_class'].value_counts()}")

if __name__ == "__main__":
    main()