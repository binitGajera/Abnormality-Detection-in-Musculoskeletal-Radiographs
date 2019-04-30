# -*- coding: utf-8 -*-
"""
@author: binit_gajera
"""
import torch

class Loss(torch.nn.modules.Module):
    def __init__(self, W1, W0):
        super(Loss, self).__init__()
        self.W1 = W1
        self.W0 = W0
        
    def forward(self, inputs, targets, phase):
        loss = - (self.W1[phase] * targets * inputs.log() + self.W0[phase] * (1 - targets) * (1 - inputs).log())
        return loss