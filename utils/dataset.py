from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2
import tifffile

#############################原始代码png
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        #print (self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask0 or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        # img = tifffile.imread(img_file[0])
        # # img = img.astype(np.int64)
        # # img = img.dtype(np.float32)
        # # img = torch.from_numpy(img)

        # mask = tifffile.imread(mask_file[0])
        # # mask = mask.astype(np.int16)
        # # mask = torch.from_numpy(mask)
        # # mask = mask.dtype(np.float32)
        # mask = np.expand_dims(mask, axis=2)
        #


        assert img.size == mask.size, \
            f'Image and mask0 {idx} should be the same size, but are {img.shape} and {mask.shape}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        # print (mask0.shape)
        return {'image': torch.from_numpy(img), 'mask0': torch.from_numpy(mask)}



# #############################适合tif的
# class BasicDataset(Dataset):
#     def __init__(self, imgs_dir, masks_dir, scale=1):
#         super(BasicDataset, self).__init__()
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
#         self.scale = scale
#
#     def __len__(self):
#         return len(self.imgs_dir)
#
#     def __getitem__(self, i):
#         mask_file = glob(self.masks_dir + '*' + str(i) + '*')
#         img_file = glob(self.imgs_dir + '*' + str(i) + '*')
#
#         # print("img_file:", img_file)
#         # print("mask_file:", mask_file)
#         if len(img_file) > 0:
#             img = tifffile.imread(img_file[0])
#         else:
#             print("Error: img_file list is empty.")
#
#         img = tifffile.imread(img_file[0])
#         img_array = np.array(img, dtype=np.float32) / 65535
#         mask = tifffile.imread(mask_file[0])
#         mask = np.array(mask, dtype=np.float32)
#         img_array = np.transpose(img_array, (2, 0, 1))
#         mask = np.expand_dims(mask, axis=2)
#         mask = np.transpose(mask, (2, 0, 1))
#
#         return {
#             'image': torch.from_numpy(img_array),
#             'mask0': torch.from_numpy(mask),
#             'img_file': img_file[0],
#             'mask_file': mask_file[0]
#         }
#
#
#
#
#
#
#     # # def __getitem__(self, idx):
#     #     img = tifffile.imread(self.imgs_dir[idx])
#     #     img_array = np.array(img, dtype=np.float32) / 65535   #归一化到0-1
#     #     mask = tifffile.imread(self.masks_dir[idx])
#     #     mask = np.array(mask, dtype=np.float32)
#     #     img_array = np.transpose(img_array, (2, 0, 1))
#     #     mask = np.expand_dims(mask, axis=2)         #对mask加一个水体分类波段，与image对应
#     #     mask = np.transpose(mask, (2, 0, 1))
#     #
#     #     return {'image': torch.from_numpy(img_array), 'mask0': torch.from_numpy(mask)}
#
