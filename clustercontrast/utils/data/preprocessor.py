from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
from clustercontrast.utils.data import transforms as T
class ChannelExchange_prompt(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """

    def __init__(self, gray = 2):
        self.idx = gray

    def __call__(self, img):
        # idx = random.randint(0, self.gray)
        if self.idx ==0:
            # random select R Channel
            img[1, :,:] = img[self.idx,:,:]
            img[2, :,:] = img[self.idx,:,:]
        elif self.idx ==1:
            # random select B Channel
            img[0, :,:] = img[self.idx,:,:]
            img[2, :,:] = img[self.idx,:,:]
        elif self.idx ==2:
            # random select G Channel
            img[0, :,:] = img[self.idx,:,:]
            img[1, :,:] = img[self.idx,:,:]
        else:
            tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
            img[0,:,:] = tmp_img
            img[1,:,:] = tmp_img
            img[2,:,:] = tmp_img
        return img

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, indices):
        return self._get_single_item(indices)
    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, index

class Preprocessor_ir(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor_ir, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        # self.trans_prompt0 = T.Compose([ChannelExchange_prompt(gray =0)])
        # self.trans_prompt1 = T.Compose([ChannelExchange_prompt(gray =1)])
        # self.trans_prompt2 = T.Compose([ChannelExchange_prompt(gray =2)])
        # self.trans_prompt3 = T.Compose([ChannelExchange_prompt(gray =3)])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, indices):
        return self._get_single_item(indices)
    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img_ori = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img_ori)
            # img2 = self.trans_prompt0(img)
            # img3 = self.trans_prompt1(img)
            # img4 = self.trans_prompt2(img)
            # img5 = self.trans_prompt3(img)
        return img,fname, pid, camid, index

class Preprocessor_color(Dataset):
    def __init__(self, dataset, root=None, transform=None,transform1=None):
        super(Preprocessor_color, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform1 = transform1
        # self.trans_prompt0 = T.Compose([ChannelExchange_prompt(gray =0)])
        # self.trans_prompt1 = T.Compose([ChannelExchange_prompt(gray =1)])
        # self.trans_prompt2 = T.Compose([ChannelExchange_prompt(gray =2)])
        # self.trans_prompt3 = T.Compose([ChannelExchange_prompt(gray =3)])
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_ori = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img_ori)
            img1 = self.transform1(img_ori)
            # img2 = self.trans_prompt0(img)
            # img3 = self.trans_prompt1(img)
            # img4 = self.trans_prompt2(img)
            # img5 = self.trans_prompt3(img)

        return img, img1,fname, pid, camid, index
