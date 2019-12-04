
#mri_input_filename = os.path.join(data_dir,'C:/Users/Ali ktk/.spyder-py3/dataloader/data/0.nii.gz')
#
#mri_gt_filename = os.path.join(data_dir,'C:/Users/Ali ktk/.spyder-py3/dataloader/data/0.nii.gz')
#
#pair = mt_datasets.SegmentationPair2D(mri_input_filename, mri_gt_filename)
#slice_pair = pair.get_pair_slice(0)
#input_slice = slice_pair["input"]
#gt_slice = slice_pair["gt"]

#with h5py.File('C:/Users/Ali ktk/.spyder-py3/dataloader/hdf5_data.h5', 'r') as hdf:
#ls=list(hdf.keys())
#prit('List of datasets in the file: \n' , ls)
#data= hdf.get("dataset1')
#dataset1= np.array(data)
#print('Shape of dataset1: \n' , dataset1.shape)
#


import nibabel as nib
from nibabel.analyze import AnalyzeImage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import PIL.Image as Image
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from nilearn import image
import cv2
import pandas as pd
import torchvision
import torchvision.transforms as tfms
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy as sc
import os
import PIL
import PIL.Image as Image
import seaborn as sns
import warnings
import h5py
import torch.utils.data as data
import glob
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from torchvision.transforms import Compose
import zipfile
from io import BytesIO
import pickle
from itertools import chain
from time import time
from fnmatch import fnmatch
from multiprocessing import Pool
from argparse import ArgumentParser



root_dir = 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1'
img_dir = 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/2'






#data_dir='C:/Users/Ali ktk/.spyder-py3/CNN detection of AD/dataset/mri/train/1'                                             #image directory
#img=nib.load(os.path.join(data_dir,'ADNI_137_S_1414_MR_MPR__GradWarp__N3__Scaled_Br_20100113101118651_S72806_I163393.nii'))                           #loading the image
#img_data=img.get_data()                                                     #accessing image array
#multi_slice_viewer(img_data)
#plt.show()





class data(Dataset):
    def __init__(self, img_dir, crop_size=192, voxel_slices=16, shuffle=False):
        self.patients = glob(img_dir)
        self.crop_size = crop_size
        self.slices = 128           # how many slices to take from 155 slice volume
        self.voxel_slices = voxel_slices
        
    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        vol = np.zeros((1, 240, 240,155))

        path = glob(self.patients[index] + 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/*')
        vol = nib.load(path[0]).get_data()
        vol = np.swapaxes(vol,-1,0)  # (240,240,155) -> (155,240,240)        
        if self.crop_size is not None:
            start = vol.shape[-1]//2 - self.crop_size//2
            stop = vol.shape[-1]//2 + self.crop_size//2
            num_chunks = self.slices//self.voxel_slices
            
            #remove blank channels (0-8 and 136-155) and crop to crop_size
            vol = vol[8: 8 + self.slices, start:stop, start:stop]
              
        voxels = np.zeros((num_chunks, self.voxel_slices, self.crop_size, self.crop_size))     # 1, 128//16, 16, 192, 192
            
        for i in range(num_chunks):
            voxels[i,:,:,:] =  vol[i*self.voxel_slices : (i+1)*self.voxel_slices, :, :]
                
        _voxels = torch.from_numpy(voxels).float()    
        _gt = torch.from_numpy(voxels[-1]).long()
        print(_voxels.shape)
        return _voxels, _gt
















#bs = 2
#num_epochs = 100
#learning_rate = 1e-3
#mom  = 0.9
#
#class Dataloder_img(data.Dataset):
#    def __init__(self,root_dir,seg_dir,transforms ):
#        self.root_dir = root_dir
#        self.seg_dir = seg_dir
#        self.transforms = transforms
#        self.files = os.listdir(self.root_dir)
#        self.lables = os.listdir(self.seg_dir)
#        print(self.files)
#    
#    def __len__(self):
#        return len(self.files)
#    
#    def __getitem__(self,idx):
#        img_name = self.files[idx]
#        label_name = self.lables[idx]
#        img = nib.load(os.path.join(self.root_dir,img_name)) #!Image.open(os.path.join(self.root_dir,img_name))
#        #change to numpy
#        img = np.array(img.dataobj)
#        #change to PIL 
#        img = Image.fromarray(img.astype('uint8'), 'RGB')
#        
#        print(img.size)
#        
#        label = nib.load(os.path.join(self.seg_dir,label_name))#!Image.open(os.path.join(self.seg_dir,label_name))
#        #change to numpy
#        label = np.array(label.dataobj)
#        #change to PIL 
#        label = Image.fromarray(label.astype('uint8'), 'RGB')
#        
#        print(label.size)
#        
#        if self.transforms:
#            img = self.transforms(img)
#            label = self.transforms(label)
#            return img,label
#        else:
#            return img, label
#full_dataset = Dataloder_img(' image ',
#                                     ' labels ',tfms.Compose([tfms.RandomRotation(180),tfms.ToTensor()
#                                                            ]))#
#                                   
#
#train_size = int(0.8 * len(full_dataset))
#val_size = len(full_dataset) - train_size
#train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
#train_loader = data.DataLoader(train_dataset,shuffle=False,batch_size=bs)
#val_loader = data.DataLoader(val_dataset,shuffle=False,batch_size=bs)
#
#test_img, test_lb = next(iter(full_dataset))
#print(test_img[0].shape)
#plt.imshow(test_img[0])
#plt.show()





















