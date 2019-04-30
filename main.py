# -*- coding: utf-8 -*-
"""
Abnormality Detection in Musculoskeletal Radiographs
@author: binit_gajera
"""

#To ignore some unwanted warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#pandas is used to load the Dataset
import pandas as pd
import os


from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from image_loading import Mura
from loss import Loss
from utils import get_count, np_V, train_model

torch.cuda.is_available()

data_cat = ['train', 'valid'] # data categories

mura_dataset = {}
study_label = {'positive': 1, 'negative': 0}
for phase in data_cat:
    BASE_DIR = './data/%s/%s/' % (phase, 'XR_HAND')
    patients = list(os.walk(BASE_DIR))[0][1] #list of patient folder names
    mura_dataset[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
    i = 0
    for patient in tqdm(patients): #per patient folder
        for study in os.listdir(BASE_DIR + patient): # per study in that patient folder
            label = study_label[study.split('_')[1]] # get label 0 or 1
            path = BASE_DIR + patient + '/' + study + '/' # path to a study
            mura_dataset[phase].loc[i] = [path, len(os.listdir(path)), label] # add new row
            i+=1

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: Mura(mura_dataset[x], transform=data_transforms[x]) for x in data_cat}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=4) for x in data_cat}

dataset_sizes = {x: len(mura_dataset[x]) for x in data_cat}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ta = total abnormal images, tn = total normal images
ta = {x: get_count(mura_dataset[x], 'positive') for x in data_cat}
tn = {x: get_count(mura_dataset[x], 'negative') for x in data_cat}
W1 = {x: np_V(tn[x] / (tn[x] + ta[x])) for x in data_cat}
W0 = {x: np_V(ta[x] / (tn[x] + ta[x])) for x in data_cat}

base_model=models.densenet169(pretrained=True)

for param in base_model.parameters():
    param.requires_grad = False

in_feat=base_model.classifier.out_features

#Adding a linear layer that produces only 1 class output and applying Sigmoid over it
model=nn.Sequential(
    base_model,
    nn.Linear(in_feat,1),
    nn.Sigmoid()
)

model=model.to(device)

criterion=Loss(W1, W0)
optimizer=optim.Adam(params=model.parameters(),lr=0.0001)
dynamic_lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

model = train_model(model, criterion, optimizer, dataloaders, dynamic_lr_scheduler, dataset_sizes, num_epochs=10)