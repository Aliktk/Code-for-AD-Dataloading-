import torch
import torch.nn as nn
import numpy as np
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
import nibabel as nib

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
import h5py
import torch.utils.data as data
import glob
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from torchvision.transforms import Compose
from collections import Counter



root_dir = 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1'
seg_dir = 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/test/1'

#def get_sub_folders(folder):
#    return [sub_folder for sub_folder in os.listdir(folder) if os.path.isdir(os.path.join(folder, sub_folder))]
#print(get_sub_folders)
#
#def get_image_type_from_folder_name(folder_name):
#    image_types = ['.nii', '.nii.gz']
#    return next(image_type for image_type in image_types if image_type in folder_name)
#print(get_image_type_from_folder_name)
#
#
#def get_extension(filename):
#    filename, extension = os.path.splitext(filename)
#    return extension
#print(get_extension)
#
#
#
#
#def analyse_data(input_dir):
#
#    shapes = []
#    relative_volumes = []
#    for folder in get_sub_folders(input_dir):
#        print(folder)
#        for sub_folder in get_sub_folders(os.path.join(input_dir, folder)):
#
#            image_type = get_image_type_from_folder_name(sub_folder)
#
#            # do not save the raw data (too heavy)
#            if image_type != '.OT':
#                continue
#
#            path = os.path.join(input_dir, folder, sub_folder)
#            filename = next(filename for filename in os.listdir(path) if get_extension(filename) == '.nii')
#            path = os.path.join(path, filename)
#            im = nib.load(path)
#            image = im.get_data()
#            shape = image.shape
#            shapes.append(shape)
#            relative_volumes.append(100 * np.sum(image) / np.cumprod(shape)[-1])
#plt.show(image)
##return shapes, relative_volumes


class Dataloder_img(data.Dataset):
    def __init__(self,root_dir,seg_dir,transforms ):
        self.root_dir = root_dir
        self.seg_dir = seg_dir
        self.transforms = transforms
        self.files = os.listdir(self.root_dir)
        self.lables = os.listdir(self.seg_dir)
        print(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        img_name = self.files[idx]
        label_name = self.lables[idx]
        img = nib.load(os.path.join(self.root_dir,img_name)) #!Image.open(os.path.join(self.root_dir,img_name))
        #change to numpy
        img = np.array(img.dataobj)
        #change to PIL 
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        
        print(img.size)
        
        label = nib.load(os.path.join(self.seg_dir,'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/ADNI_136_S_0300_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080529142830882_S50401_I107759.nii'))#!Image.open(os.path.join(self.seg_dir,label_name))
        #change to numpy
        label = np.array(label.dataobj)
        #change to PIL 
        label = Image.fromarray(label.astype('uint8'), 'RGB')
        
        print(label.size)
        
        if self.transforms:
            img = self.transforms(img)
            label = self.transforms(label)
            return img,label
        else:
            return img, label
full_dataset = Dataloder_img('C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1',
                                     'C:/Users/Ali ktk/.spyder-py3/dataloader/data/test/1',tfms.Compose([tfms.RandomRotation(0),tfms.Resize((256,256)),tfms.ToTensor()
                                                            ]))#
                                   

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader = data.DataLoader(train_dataset,shuffle=False,batch_size=4)
val_loader = data.DataLoader(val_dataset,shuffle=False,batch_size=4)
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
test_img, test_lb = next(iter(full_dataset))
print(test_img[0].shape)
plt.imshow(test_lb[0])












