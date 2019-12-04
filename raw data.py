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
from scipy import ndimage
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

cwd = 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1'
#data_dir = 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/test/0/ADNI_137_S_0800_MR_MPR__GradWarp__N3__Scaled_Br_20081024125416989_S56178_I123506.nii'


#cwd = os.getcwd()#return current working directory
print(cwd)
#debugger
patient_id = os.listdir('C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1') #create a list that encompasses all the images we downloaded
len(patient_id)
print(patient_id)
#



#filename = glob.glob('C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/ADNI_137_S_0800_MR_MPR__GradWarp__N3__Scaled_Br_20081024125416989_S56178_I123506.nii' + 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/ADNI_137_S_0800_MR_MPR__GradWarp__N3__Scaled_Br_20081024125416989_S56178_I123506.nii')
#print(filename)    


#myfiles = [i for i in os.listdir(".") if i.endswith(".nii")]
#
#for image in range(len(patient_id)):
#    loaded_image = nib.load(os.path.abspath(myfiles[image]))

dataimage = [patient_id]#[patient_id, image_matrix_normalized]
for patient in patient_id:
    for root, dirs, files in os.walk(cwd + "ADNI" + patient):
        flag = 0 
        for file in files:
            if file.endswith(".nii"):
                print(os.path.join(root, file))
                if flag < 1:
                    datapath.append(os.path.join(root, file))
                    print(os.path.join(root, file))
                    print(flag)
                    flag = flag + 1
datapath = patient_id
#
##tar_dim = [256, 256, 160]
#
for path in datapath:
    mri = nib.load(path).get_data()
    mri = (mri - mri.min())/(mri.max() - mri.min())
    mri = mri - mri.mean()
    tar_mri = ndimage.zoom(mri,[tar_dim[0]/mri.shape[0],tar_dim[1]/mri.shape[1], tar_dim[2]/mri.shape[2]], order = 1)

    print(mri)
    print(mri.mean())
    print(mri.var())
    print(tar_mri.shape)
    plt.imshow(mri[125,:,:],'gray')
    plt.show()
    
    dataimage.append(tar_mri)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    