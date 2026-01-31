import os
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


class PairedSRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale=4, hr_crop=128, train=True):
        self.pairs = self._build_pairs(lr_dir, hr_dir)
        self.scale = scale
        self.hr_crop = hr_crop
        self.train = train

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _nat_key(path):
        name = os.path.basename(path)
        parts = re.split(r'(\d+)', name)
        key = []
        for p in parts:
            if p.isdigit():
                key.append(int(p))
            else:
                key.append(p.lower())
        return key

    @staticmethod
    def _pair_key(path):
        name = os.path.splitext(os.path.basename(path))[0].lower()
        name = re.sub(r'(?:_?lr|_?hr)$', '', name)
        digits = re.findall(r'\d+', name)
        return digits[0] if digits else name

    def _build_pairs(self, lr_dir, hr_dir):
        lr_files = [
            os.path.join(lr_dir, f) for f in os.listdir(lr_dir)
            if os.path.isfile(os.path.join(lr_dir, f))
        ]
        hr_files = [
            os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
            if os.path.isfile(os.path.join(hr_dir, f))
        ]

        lr_map = {self._pair_key(p): p for p in lr_files}
        hr_map = {self._pair_key(p): p for p in hr_files}
        keys = sorted(set(lr_map) & set(hr_map), key=self._nat_key)
        pairs = [(lr_map[k], hr_map[k]) for k in keys]
        if not pairs:
            raise ValueError('No matched LR/HR pairs found.')
        return pairs

    def _load(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _paired_random_crop(self, lr_img, hr_img):
        hr_w, hr_h = hr_img.size
        if hr_w < self.hr_crop or hr_h < self.hr_crop:
            raise ValueError('HR image smaller than crop size')

        hr_left = random.randint(0, hr_w - self.hr_crop)
        hr_top = random.randint(0, hr_h - self.hr_crop)

        lr_crop = self.hr_crop // self.scale
        lr_left = hr_left // self.scale
        lr_top = hr_top // self.scale

        hr_patch = TF.crop(hr_img, hr_top, hr_left, self.hr_crop, self.hr_crop)
        lr_patch = TF.crop(lr_img, lr_top, lr_left, lr_crop, lr_crop)
        return lr_patch, hr_patch

    def _paired_augment(self, lr, hr):
        # Random Horizontal Flip
        if random.random() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        # Random Vertical Flip
        if random.random() < 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)
        # Random Rotate 90
        if random.random() < 0.5:
            lr = TF.rotate(lr, 90)
            hr = TF.rotate(hr, 90)
        return lr, hr

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        lr_img = self._load(lr_path)
        hr_img = self._load(hr_path)

        if self.train:
            lr_img, hr_img = self._paired_random_crop(lr_img, hr_img)
            lr_img, hr_img = self._paired_augment(lr_img, hr_img)

        lr_tensor = TF.to_tensor(lr_img)
        hr_tensor = TF.to_tensor(hr_img)
        return lr_tensor, hr_tensor


def build_loader(lr_dir, hr_dir, scale, hr_crop, batch_size, num_workers, train):
    dataset = PairedSRDataset(lr_dir, hr_dir, scale=scale, hr_crop=hr_crop, train=train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader
