import os
import glob
import cv2
import random
import torch
from loguru import logger
from torch.utils.data import Dataset
import numpy as np

def get_patch(hr_img, lr_img, patch_size=96, scale=4):
    lh, lw = lr_img.shape[0], lr_img.shape[1]

    lr_patch_size = patch_size // scale
    lx = random.randrange(0, lw-lr_patch_size+1)
    ly = random.randrange(0, lh-lr_patch_size+1)
    _lr_img_patch = lr_img[ly: ly+lr_patch_size, lx: lx+lr_patch_size, :]

    hr_patch_size = lr_patch_size * scale
    hx = lx * scale
    hy = ly * scale
    _hr_img_patch = hr_img[hy: hy+hr_patch_size, hx: hx+hr_patch_size, :]

    return {'hr_patch': _hr_img_patch, 'lr_patch': _lr_img_patch}

def augment(pair, hflip=True, vflip=True, rot90=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot90 and random.random() < 0.5
    rot90 = rot90 and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    pair['hr_patch'] = _augment(pair['hr_patch'])
    pair['lr_patch'] = _augment(pair['lr_patch'])
    return pair

class SRDataset(Dataset):
    def __init__(self, lr_path, hr_path, patch_size=96, scale=4, aug=False, normalization=0, need_patch=False, suffix='png'):

        self.patch_size = patch_size
        self.scale = scale
        self.aug = aug
        self.normalization = normalization

        self.lr_path = lr_path
        self.hr_path = hr_path
        self.need_patch = need_patch

        self.hr_lists = glob.glob(hr_path + "*." + suffix)
        self.lr_lists = glob.glob(lr_path + "*." + suffix)
        self.hr_lists.sort()
        self.lr_lists.sort()

        self.len = len(self.hr_lists)
        assert self.len == len(self.lr_lists)
        logger.info(f'Find {self.len} images in {self.hr_path} and {self.lr_path} respectively.')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        hr_img = cv2.imread(self.hr_lists[idx])
        lr_img = cv2.imread(self.lr_lists[idx])
        file_name = os.path.basename(self.hr_lists[idx]).split('.')[0]

        if self.need_patch:
            patch_pair = get_patch(hr_img, lr_img, self.patch_size, self.scale)
        else:
            patch_pair = {'hr_patch': hr_img, 'lr_patch': lr_img}

        # augment the dataset
        if self.aug:
            patch_pair = augment(patch_pair)

        # normalization
        if self.normalization == 0:
            pass
        elif self.normalization == 1:
            patch_pair['hr_patch'] = patch_pair['hr_patch'] / 255.0
            patch_pair['lr_patch'] = patch_pair['lr_patch'] / 255.0
        else:
            raise NotImplementedError

        patch_pair['hr_patch'] = np.transpose(patch_pair['hr_patch'], (2, 0, 1)).astype(np.float32)
        patch_pair['lr_patch'] = np.transpose(patch_pair['lr_patch'], (2, 0, 1)).astype(np.float32)

        patch_pair['hr_patch'] = torch.from_numpy(patch_pair['hr_patch'])
        patch_pair['lr_patch'] = torch.from_numpy(patch_pair['lr_patch'])

        return {'hr': patch_pair['hr_patch'], 'lr': patch_pair['lr_patch'], 'fn': file_name}

if __name__ == "__main__":
    hr_path, lr_path = 'D:/Dataset/DIV2K/DIV2K_train_HR/', 'D:/Dataset/DIV2K/DIV2K_train_LR_x4d/'
    dataset_ = SRDataset(lr_path, hr_path, need_patch=True)
    data = dataset_[0]
    print(f'data[\'hr\'].shape: {data["hr"].shape}\n'
          f'data[\'lr\'].shape: {data["lr"].shape}\n'
          f'data[\'fn\']: {data["fn"]}\n')