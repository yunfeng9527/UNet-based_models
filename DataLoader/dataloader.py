import logging
import random

import numpy as np
import torch
from PIL import Image

from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF
import torchvision.transforms as T


def load_image(filename):
    """加载图像或掩码文件"""
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    """统计掩码文件中的唯一颜色值"""
    mask_files = list(mask_dir.glob(idx + mask_suffix + '.*'))
    assert mask_files, f"找不到掩码文件: {idx + mask_suffix + '.*'}"

    mask_file = mask_files[0]
    mask = np.asarray(load_image(mask_file).convert('RGB'))  # 强制转换成 RGB

    unique_colors = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0)
    return [tuple(c) for c in unique_colors]


class BasicDataset(Dataset):
    # COLOR_TO_INDEX = {
    #     (0, 0, 0): 0,
    #
    # }

    def __init__(self, images_dir: str, mask_dir: str, num_classes: int,  mask_suffix: str = '',transform = None,
        joint_transform = None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)


        self.num_classes = num_classes
        self.mask_suffix = mask_suffix
        self.transform = transform  # 图像增强
        self.joint_transform = joint_transform  # 联合增强（图+mask）

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'在 {images_dir} 目录下未找到任何输入图像')

        logging.info(f'创建数据集，共 {len(self.ids)} 个样本')
        logging.info('扫描掩码文件以确定唯一值')

        # self.mask_values = list(self.COLOR_TO_INDEX.keys())


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, is_mask):

        """预处理输入图像和掩码，调整为 统一大小"""
        target_size = (512, 512)  # 宽 × 高

        if pil_img.size != target_size:
            pil_img = pil_img.resize(target_size, resample=Image.NEAREST if is_mask else Image.BICUBIC)

        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((target_size[1], target_size[0]), dtype=np.int64)
            if img.ndim == 2:  # 已经是类别索引
                return img
            elif img.ndim == 3:  # RGB 掩码
                for color, idx in BasicDataset.COLOR_TO_INDEX.items():
                    mask[(img == color).all(axis=-1)] = idx
                return mask
            else:
                raise ValueError(f"掩码格式不符合预期: {img.shape}")

        else:  # 预处理图像
            if img.ndim == 2:
                img = img[np.newaxis, ...]  # 灰度图
            else:
                img = img.transpose((2, 0, 1))  # HWC -> CHW

            if (img > 1).any():
                img = img / 255.0  # 归一化

            return img

    def __getitem__(self, idx):
        """加载单个样本"""
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'ID {name} 对应多个或没有图像: {img_file}'
        assert len(mask_file) == 1, f'ID {name} 对应多个或没有掩码: {mask_file}'

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])


        assert img.size == mask.size, \
            f'图像和掩码尺寸不匹配: {name}, 图像 {img.size}, 掩码 {mask.size}'

        if self.joint_transform:
            img, mask = self.joint_transform(img, mask)

            # 图像独立 transform
        if self.transform:
            img = self.transform(img)

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)


        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),

        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1,transform = None,
        joint_transform = None):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='', transform=transform, joint_transform=joint_transform)



class JointTransform:
    def __init__(self):
        pass

    def __call__(self, img, mask):
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(scale, scale), ratio=(1.0, 1.0))
            img = TF.resized_crop(img, i, j, h, w, size=(512, 512), interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, size=(512, 512), interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

img_only_transform = T.Compose([
    T.GaussianBlur(3),
    T.ColorJitter(brightness=0.2, contrast=0.2),
])



