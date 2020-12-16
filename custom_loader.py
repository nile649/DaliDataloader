from torchvision import transforms
from torch.autograd import Variable


import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
import collections
import random
from scipy import ndimage

#custom function for 4D operation.

class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


    
class Transform(object):
    def __init__(self,norm=True):
        self.norm = norm
    def test_transform(self):
        transform_list = []
#         transform_list.append(transforms.ToPILImage())
        transform_list.append(ToTensor())
        if self.norm:
            transform_list.append(Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.25, 0.25, 0.25]))
 
        transform = transforms.Compose(transform_list)
        return transform#.unsqueeze(0)
    def __call__(self):
        return self.test_transform()
    

def image_loader(imgarray,arg):
    
    """load image, returns cuda tensor"""
    transform = Transform(arg.norm)
    transform_ = transform()
    image = transform_(imgarray)
    image = Variable(image, requires_grad=True)
#     image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    if arg.device=='cuda':
        return image.cuda()  #assumes that you're using GPU
    else:
        return image
