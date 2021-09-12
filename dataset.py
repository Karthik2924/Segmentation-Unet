# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 00:40:51 2021

@author: Karthik
"""
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class custom_dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = os.listdir(img_dir)

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir,self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask