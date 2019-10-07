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
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from nilearn import image
import cv2
import h5py
import copy
import string
import gzip
import torch.utils.data as data
from scipy import ndimage
from scipy import linalg
import os
from glob import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from torchvision.transforms import Compose
import zipfile
from io import BytesIO
import pickle

#batch_size = 10

#data_nii = glob.glob('C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/*')
cwd = 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/*'


cwd = os.getcwd()#return current working directory
patient_id = os.listdir('C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/') #create a list that encompasses all the images we downloaded
print(cwd)
print(patient_id)

#filename = glob.glob('C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/' + '*/data_batch*')


patient_num = 6
datapath = [patient_id, 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/'] #[patient_id, path]
#datapath part
for patient in patient_id:
    flag = 0
    for root, dirs, files in os.walk(cwd + "/ADNI/" + patient):       
        for file in files:
            if file.endswith(".nii"):
                if flag < 1:
                    print([patient, os.path.join(root, file)])####
                    datapath.append([patient, os.path.join(root, file)])
                    flag = flag + 1

dataimage = [patient_id]#[patient_id, image_matrix_normalized]


tar_dim = [256, 256, 160]

for path in datapath:
    mri = nib.load(path[1]).get_data()
    mri = (mri - mri.min())/(mri.max() - mri.min())
    mri = mri - mri.mean()
    tar_mri = ndimage.zoom(mri,[tar_dim[0]/mri.shape[0],tar_dim[1]/mri.shape[1], tar_dim[2]/mri.shape[2]], order = 1)

    #print(mri)
    #print(mri.mean())
    #print(mri.var())
    #print(tar_mri.shape)
    plt.imshow(mri[125,:,:],'gray')
    plt.show()
    print([path[0],tar_mri.shape])
 
    dataimage.append([path[0],tar_mri])

 
print(len(dataimage))

with open("dataimage_arr.txt", "wb") as fp:
    pickle.dump(dataimage, fp)
with open("datacsv_new_arr.txt", "wb") as fp:
    pickle.dump(datacsv_new, fp)






































#img.affine
#np.set_printoptions(precision=2, suppress=True)
#img.dataobj
#img.header
#header = img.header
#print(header) 
#print(header.get_data_shape())
#print(header.get_data_dtype())
#print(header.get_zooms())
#img.dataobj
#nib.is_proxy(img.dataobj)
#array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
#affine = np.diag([1, 2, 3, 1])
#array_img = nib.Nifti1Image(array_data, affine)
#array_img.dataobj is array_data
#image_data = array_img.get_fdata()
#image_data.shape
#image_data.dtype == np.dtype(np.float64)








#img.shape
#img.get_data_dtype() == np.dtype(np.int16)
#img.affine.shape
#data = img.get_fdata()    
#data.shape
#type(data)
#hdr = img.header
#hdr.get_xyzt_units()
#
#raw = hdr.structarr
#
#raw['xyzt_units']
#
#
#
#data = np.ones((32, 32, 15, 100), dtype=np.int16)
#img = nib.Nifti1Image(data, np.eye(4))
#img.get_data_dtype() == np.dtype(np.int16)
#img.header.get_xyzt_units()
#
#nib.save(img, os.path.join('C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1','test4d.nii.gz'))
#
#img1 = nib.load('C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/ADNI_941_S_1311_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080703170241434_S51039_I112538.nii')
#
#
#data = img1.get_data()
#affine = img1.affine
#
#print(img1)
#
#nib.save(img1, 'my_file_copy.nii.gz')
#
#new_image = nib.Nifti1Image(data, affine)
#nib.save(new_image, 'new_image.nii.gz')




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 2, 5)
        self.conv2 = nn.Conv3d(2, 4, 5)
        self.conv3 = nn.Conv3d(4, 8, 5)
        self.conv4 = nn.Conv3d(8, 16, 5)
        self.conv5 = nn.Conv3d(16, 32, 5)
        self.pool = nn.MaxPool3d(2, 2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8 * 5, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x): #256*256*160*1
        x = self.pool(F.relu(self.conv1(x)))#batch_size * 128 * 128 * 80 * 2
        x = self.pool(F.relu(self.conv2(x)))#batch_size * 64 * 64 * 40 * 4
        x = self.pool(F.relu(self.conv3(x)))#batch_size * 32 * 32 * 20 * 8
        x = self.pool(F.relu(self.conv4(x)))#batch_size * 16 * 16 * 10 * 16
        x = self.pool(F.relu(self.conv5(x)))#batch_size * 8 * 8 * 5 * 32
        x = x.view(-1, 32 * 8 * 8 * 5)#batch_size * 10240(8*8*5*32)
        x = F.relu(self.fc1(x))#batch_size * 1024
        x = F.relu(self.fc2(x))#batch_size * 256
        x = F.relu(self.fc3(x))#batch_size * 64
        x = self.fc4(x)#batch_size * 3
        return x


net = Net()















































































































#nii_files = []
#for data_dir, nii_files, files in os.walk('My data'):
#    if 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1' in files:
#        nii_files.append(os.path.join(dirpath,  'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/ADNI1_Annual_2_Yr_1.5T.nii.gz'))
##ADNI1_Annual_2_Yr_1.5T.nii.gz
#for i in nii_files:
#    decompressed_file = gzip.open(i)
#    out_path = i.replace('/','_')[:-3]
#    with open('C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1/ADNI1_Annual_2_Yr_1.5T.nii.gz' + out_path, 'wb') as outfile:
#        outfile.write(decompressed_file.read())



#dirs = 'C:/Users/Ali ktk/.spyder-py3/dataloader/data'
#
#indir = 'C:/Users/Ali ktk/.spyder-py3/dataloader/data/train/1'
#Xs = []
#for root, dirs, filenames in os.walk(indir):
#    for f in filenames:
#        if '.nii' == f[-4:]:
#            img = nib.load(indir + f)
#            data = img.dataobj # Get the data object
#            data = data[:-1,:-1,:-1] # Clean the last dimension for a high GCD (all values are 0)
#            X = np.expand_dims(data, -1)
#            X = X / np.max(X)
#            X = X.astype('float32')
#            X = np.expand_dims(X, 0)
#            print('Shape: ', X.shape)
#            Xs.append(X)
#            
#Xa = np.vstack(Xs)
#save_large_dataset('Xa', Xa)
























