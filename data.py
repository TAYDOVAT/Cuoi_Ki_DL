import os
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


class PairedSRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale=4, hr_crop=128, train=True):
        self.lr_paths = sorted([
            os.path.join(lr_dir, f) for f in os.listdir(lr_dir)
            if os.path.isfile(os.path.join(lr_dir, f))
        ], key=self._nat_key)
        self.hr_paths = sorted([
            os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
            if os.path.isfile(os.path.join(hr_dir, f))
        ], key=self._nat_key)
        self.scale = scale
        self.hr_crop = hr_crop
        self.train = train

    def __len__(self):
        return min(len(self.lr_paths), len(self.hr_paths))

    @staticmethod
    def _nat_key(path):
        name = os.path.basename(path)
        parts = re.split(r'(\\d+)', name)
        key = []
        for p in parts:
            if p.isdigit():
                key.append(int(p))
            else:
                key.append(p.lower())
        return key

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

    def __getitem__(self, idx):
        lr_img = self._load(self.lr_paths[idx])
        hr_img = self._load(self.hr_paths[idx])

        if self.train:
            lr_img, hr_img = self._paired_random_crop(lr_img, hr_img)

        lr_tensor = TF.to_tensor(lr_img)
        hr_tensor = TF.to_tensor(hr_img)
        return lr_tensor, hr_tensor


def build_loader(lr_dir, hr_dir, scale, hr_crop, batch_size, num_workers, train):
    dataset = PairedSRDataset(lr_dir, hr_dir, scale=scale, hr_crop=hr_crop, train=train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    return dataset, loader
