import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
import collections
import random
from scipy import ndimage
import pdb


# Borht have class Transform, we return transform object after initialization.
class normalize(object):
    ''' 
    The Retina face requires the following normalization value for the pre-trained model.
    '''
    def __call__(self, img):
        img -= [104.0, 117.0, 123.0]
        return img

    def __repr__(self):
        return self.__class__.__name__+'()'
    
class Transform(object):
    def __init__(self,usePIL=True):
        self.usePIL = usePIL
    def test_transform(self):
        transform_list = []
        if self.usePIL==False:
            transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.Resize((750,750)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.407, 0.46, 0.48), (1.0,1.0,1.0)))
        transform = transforms.Compose(transform_list)
        
        return transform#.unsqueeze(0)
    def __call__(self):
        return self.test_transform()
    
class Transform3(object):
    def __init__(self,norm=True):
        self.norm = norm
    def test_transform(self):
        transform_list = []
        if self.norm:
            transform_list.append(normalize())
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform#.unsqueeze(0)
    def __call__(self):
        return self.test_transform()
    
class Transform2(object):
    def __init__(self,usePIL=True):
        self.usePIL = usePIL
    def test_transform(self):
        transform_list = []
        if self.usePIL==False:
            transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform#.unsqueeze(0)
    def __call__(self):
        return self.test_transform()