import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
# from sklearn.preprocessing import MinMaxScaler
from os.path import join
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
import os
import sys
import argparse
import time
import math
import pandas as pd
from sklearn.model_selection import KFold
import cv2
from torchvision import transforms
from scipy import ndimage
import nibabel as nib

def add_salt_peper_3D(image, amout):
    s_vs_p = 0.5
    noisy_img = np.copy(image)
    num_salt = np.ceil(amout * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0], coords[1]] = 1.
    num_pepper = np.ceil(amout * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0], coords[1]] = 0.
    return noisy_img

def add_salt_peper(image, amout):
    s_vs_p = 0.5
    noisy_img = np.copy(image)
    num_salt = np.ceil(amout * image.shape[0] * image.shape[1] * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = 1.
    num_pepper = np.ceil(amout * image.shape[0] * image.shape[1] * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = 0.
    return noisy_img


def scale_image(image, patch_size):
    image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    return image


def resize_oct_data_trans(data, size):
    """
    Resize the data to the input size
    """
    input_D, input_H, input_W = size[0], size[1], size[2]
    data = data.squeeze()
    [depth, height, width] = data.shape
    scale = [input_D * 1.0 / depth, input_H * 1.0 / height, input_W * 1.0 / width]
    data = ndimage.interpolation.zoom(data, scale, order=0)
    # data = data.unsqueeze()
    return data




class GAMMA_dataset(Dataset):
    def __init__(self,
                 args,
                 dataset_root,
                 oct_img_size,
                 fundus_img_size,
                 mode='train',
                 label_file='',
                 filelists=None,
                 ):
        self.condition = args.condition
        self.condition_name = args.condition_name
        self.Condition_SP_Variance = args.Condition_SP_Variance
        self.Condition_G_Variance = args.Condition_G_Variance
        self.seed_idx = args.seed_idx
        self.model_base = args.model_base

        self.dataset_root = dataset_root
        self.input_D = oct_img_size[0][0]
        self.input_H = oct_img_size[0][1]
        self.input_W = oct_img_size[0][2]


        self.fundus_train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            # normalize,
        ])

        self.oct_train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ])

        self.fundus_val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.oct_val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.mode = mode.lower()
        label = {row['data']: row[1:].values
                 for _, row in pd.read_excel(label_file).iterrows()}


        self.file_list = []
        for f in filelists:
            filename = os.path.basename(f)
            if filename.isdigit():
                self.file_list.append([f, label[int(filename)]])


    def __getitem__(self, idx):
        data = dict()

        real_index, label = self.file_list[idx]


        fundus_img_path = os.path.join(self.dataset_root.replace('/MGamma/', '/multi-modality_images/'), real_index,
                                       f"{real_index}" + ".png")


        fundus_img = cv2.imread(fundus_img_path)

        # # OCT read
        oct_series_list = os.listdir(os.path.join(self.dataset_root, real_index, real_index))
        oct_images = []
        for filename in oct_series_list:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.dataset_root, real_index, real_index, filename)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('L')
                        img = np.array(img)
                        if len(img.shape) == 2:
                            img = np.expand_dims(img, axis=2)
                        oct_images.append(img)
                except IOError:
                    print(f"Cannot load {img_path}. Skipping.")
        if oct_images:
            oct_img = np.stack(oct_images, axis=0)



        if self.model_base == "transformer":
            fundus_img = scale_image(fundus_img, 384)
            oct_img = resize_oct_data_trans(oct_img, (96, 96, 96))
        else:
            fundus_img = scale_image(fundus_img, 512)
            oct_img = self.__resize_oct_data__(oct_img)

        oct_img = oct_img / 255.0
        fundus_img = fundus_img / 255.0

        np.random.seed(self.seed_idx)

        # add noise on fundus & OCT
        if self.condition == 'noise':
            if self.condition_name == "SaltPepper":
                fundus_img = add_salt_peper(fundus_img.transpose(1, 2, 0), self.Condition_SP_Variance)  # c,
                fundus_img = fundus_img.transpose(2, 0, 1)
                for i in range(oct_img.shape[0]):
                    oct_img[i, :, :] = add_salt_peper_3D(oct_img[i, :, :], self.Condition_SP_Variance)  # c,

            elif self.condition_name == "Gaussian":

                noise_add = np.random.normal(0, self.Condition_G_Variance, oct_img.shape)
                oct_img = oct_img + noise_add
                oct_img = np.clip(oct_img, 0.0, 1.0)

            else:
                noise_add = np.random.normal(0, self.Condition_G_Variance, fundus_img.shape)
                fundus_img = fundus_img + noise_add
                fundus_img = np.clip(fundus_img, 0.0, 1.0)

                noise_add = np.random.normal(0, self.Condition_G_Variance, oct_img.shape)
                oct_img = oct_img + noise_add
                oct_img = np.clip(oct_img, 0.0, 1.0)

                fundus_img = add_salt_peper(fundus_img, self.Condition_SP_Variance)  # c,

                for i in range(oct_img.shape[0]):
                    oct_img[i, :, :] = add_salt_peper_3D(oct_img[i, :, :], self.Condition_SP_Variance)  # c,

        if self.mode == "train":
            fundus_img = self.fundus_train_transforms(fundus_img.astype(np.float32))
            oct_img = self.oct_train_transforms(oct_img.astype(np.float32))
        else:
            fundus_img = self.fundus_val_transforms(fundus_img)
            oct_img = self.oct_val_transforms(oct_img)

        data[0] = fundus_img
        data[1] = oct_img.unsqueeze(0)

        label = label.argmax()
        return data, label

    def __len__(self):
        return len(self.file_list)

    def __resize_oct_data__(self, data):
        """
        Resize the data to the input size
        """
        data = data.squeeze()
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)
        # data = data.unsqueeze()
        return data



