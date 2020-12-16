import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import os
import numpy as np
from .DataTransform import Transform,Transform2,Transform3
import cv2
import torch
import copy 
from torchvision import transforms


class FlatFolderDatasetPIL(data.Dataset):
    ''' 
    Create Pytorch Custom dataloader
    '''
    def __init__(self,path):
        super(FlatFolderDatasetPIL, self).__init__()
        self.root =path
        transform1 = Transform()
        transform2 = Transform2()
        self.list = False
        if type(path)!=list:
            self.paths = os.listdir(self.root)
            self.list = False
        else:
            self.paths = self.root
            self.list = True
        self.transform1 = transform1()
        self.transform2 = transform2()
#         self.transform1 = transforms.Compose([transforms.Resize((750, 750)), \
#  \
#                                              transforms.ToTensor(), \
#                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         self.transform2 = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        path = self.paths[index]
        
        if self.list:
            image_raw = Image.open(path).convert('RGB')
        else:
            image_raw = Image.open(self.root+'/'+path).convert('RGB')
        img1 = self.transform1(image_raw)
        img2 = self.transform2(image_raw)
        return img1,img2

    def __len__(self):
        return len(self.root)

    def name(self):
        return 'FlatFolderDataset'

class FlatFolderDataset(data.Dataset):
    ''' 
    Create Pytorch Custom dataloader
    '''
    def __init__(self,path):
        super(FlatFolderDataset, self).__init__()
        self.root =path
        transform1 = Transform3(True)
        transform2 = Transform2(False)
        self.list = False
        if type(path)!=list:
            self.paths = os.listdir(self.root)
            self.list = False
        else:
            self.paths = self.root
            self.list = True
        self.transform1 = transform1()
        self.transform2 = transform2()

    def __getitem__(self, index):
        path = self.paths[index]
        if self.list:
            img_raw = cv2.imread(path, cv2.IMREAD_COLOR)
            im_path = path
        else:
            img_raw = cv2.imread(self.root+'/'+path, cv2.IMREAD_COLOR)
            im_path = self.root+'/'+path
        img = np.float32(img_raw)
        img = cv2.resize(img, (750, 750))
        img1 = self.transform1(copy.deepcopy(img))
        img2 = self.transform2(copy.deepcopy(img_raw))
        return img1,img2

    def __len__(self):
        return len(self.root)

    def name(self):
        return 'FlatFolderDataset'   
    
def dataPILCPU(path:str(),opt):
    dataset = FlatFolderDatasetPIL(path)
    return iter(data.DataLoader(
            dataset, batch_size=opt.batch_size,
            num_workers=opt.num_workers))
    print("PYTORCH INITIATED")
def dataCPU(path:str(),opt):
    dataset = FlatFolderDataset(path)
    return iter(data.DataLoader(
            dataset, batch_size=opt.batch_size,
            num_workers=opt.num_workers))
    print("PYTORCH INITIATED")