# -*- coding: utf-8 -*-
"""
@author: binit_gajera
"""
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets.folder import pil_loader

class Mura(Dataset):
    
    def __init__(self, df, transform=None):
      """
      df here would be containing the image path and labels.
      """
      self.df = df
      self.transform = transform

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
      study_p = self.df.iloc[idx, 0]
      count = self.df.iloc[idx, 1]
      images = []
      for i in range(count):
          image = pil_loader(study_p + 'image%s.png' % (i+1))
          images.append(self.transform(image))
      images = torch.stack(images)
      label = self.df.iloc[idx, 2]
      sample = {'images': images, 'label': label}
      return sample