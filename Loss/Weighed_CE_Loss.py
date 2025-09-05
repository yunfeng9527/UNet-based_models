import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

def compute_class_weights(mask_dir, num_classes, device="cuda"):

    """根据数据集掩码自动计算类别权重"""

    pixel_counts = np.zeros(num_classes, dtype=np.int64)

    # 遍历所有 mask 文件
    for file in tqdm(os.listdir(mask_dir), desc="统计类别像素数量"):
        if not (file.endswith(".png") or file.endswith(".jpg")):
            continue
        mask_path = os.path.join(mask_dir, file)
        mask = np.array(Image.open(mask_path))

        # 累计各类别像素数量
        for cls in range(num_classes):
            pixel_counts[cls] += np.sum(mask == cls)

    total_pixels = pixel_counts.sum()
    class_ratios = pixel_counts / total_pixels

    # 加权公式（可根据需求调整）
    epsilon = 1.02
    class_weights = 1.0 / np.log(epsilon + class_ratios)
    class_weights = class_weights / class_weights.sum()

    # 转成 Tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("类别像素数:", pixel_counts)
    print("类别权重:", class_weights)

    return nn.CrossEntropyLoss(weight=class_weights_tensor)
