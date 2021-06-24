#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:35:16 2021

@author: Tom
"""

import numpy as np
import scipy.io as sio
import os
import h5py
import matplotlib.pyplot as plt

# Specify filepaths
unet = input("Unet Name: ")
dataset = input("Dataset Name: ")

seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/Results/CCA_Expert1/seg/"
seg2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/Results/CCA_Expert2/seg/"
expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/Results/ExperttoExpert/expert1/"
expert2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/Results/ExperttoExpert/expert2/"
images1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Datasets/" + dataset + "_Testing_Dataset_expert1.hdf5"
images2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Datasets/" + dataset + "_Testing_Dataset_expert2.hdf5"
pt_names_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Volumes/"
out_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/" + dataset + "/"

# Get list of patient names
pt_names = os.listdir(pt_names_path)
pt_names = sorted([name.replace(".mha","") for name in pt_names])

# Get list of slices
seg1_slices = os.listdir(seg1_path)
expert1_slices = os.listdir(expert1_path)

seg2_slices = os.listdir(seg2_path)
expert2_slices = os.listdir(expert2_path)

# Read in cropped images
with h5py.File(images1_path, 'r') as f:
        images1 = f['testing_Images'][()]
        filenames1 = f['testing_image_filenames'][()]
        
images1 = images1.squeeze()
filenames1 = filenames1.tolist()
filenames1=[x.decode('utf-8') for x in filenames1]
        
with h5py.File(images2_path, 'r') as f:
    images2 = f['testing_Images'][()]
    filenames2 = f['testing_image_filenames'][()]
    
images2 = images2.squeeze()
filenames2 = filenames2.tolist()
filenames2=[x.decode('utf-8') for x in filenames2]

# Load in .mat files into array
seg1_slice_list = []
expert1_slice_list = []

seg2_slice_list = []
expert2_slice_list = []

# Load segmentation and expert slices into array

for i in range(len(seg1_slices)):
    seg1_slice = sio.loadmat(seg1_path + seg1_slices[i])
    seg1_slice = seg1_slice["prediction"]
    seg1_slice_list.append(seg1_slice)
    
    expert1_slice = sio.loadmat(expert1_path + expert1_slices[i])
    expert1_slice = expert1_slice["actual"].squeeze()
    expert1_slice_list.append(expert1_slice)
    
    seg2_slice = sio.loadmat(seg2_path + seg2_slices[i])
    seg2_slice = seg2_slice["prediction"]
    seg2_slice_list.append(seg2_slice)
    
    expert2_slice = sio.loadmat(expert2_path + expert2_slices[i])
    expert2_slice = expert2_slice["actual"]
    expert2_slice_list.append(expert2_slice.squeeze())
    
# Convert lists to arrays
seg1_slice_list_np = np.array(seg1_slice_list)
expert1_slice_list_np = np.array(expert1_slice_list)

seg2_slice_list_np = np.array(seg2_slice_list)
expert2_slice_list_np = np.array(expert2_slice_list)

# Create plots
for k in range(len(seg1_slices)):
    plt.axis('off')
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    
    ax[0].imshow(images1[k], cmap='gray', interpolation='bilinear')
    ax[0].set_title('Image')
    ax[0].grid(False)
    
    ax[1].imshow(images1[k], cmap='gray', interpolation='bilinear')
    ax[1].contour(expert1_slice_list_np[k], colors='#FF1493', levels=[1])
    ax[1].contour(expert2_slice_list_np[k], colors='c', levels=[1])
    ax[1].set_title('Image Annotated by 2 experts (Red = Expert 1 and Cyan = Expert 2')
    ax[1].grid(False)
    
    ax[2].imshow(images1[k], cmap='gray', interpolation='bilinear')
    ax[2].contour(expert1_slice_list_np[k], colors='#FF1493', levels=[1])
    ax[2].contour(seg1_slice_list_np[k], colors='#00FF00', levels=[1])
    ax[2].set_title('U-Net Compared to Expert 1')
    ax[2].grid(False)
    
    ax[3].imshow(images1[k], cmap='gray', interpolation='bilinear')
    ax[3].contour(expert2_slice_list_np[k], colors='c', levels=[1])
    ax[3].contour(seg2_slice_list_np[k], colors='#00FF00', levels=[1])
    ax[3].set_title('U-Net Compared to Expert 2')
    ax[3].grid(False)
    
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    ax[3].set_axis_off()
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight')
    
    
    
# Select a patient
# pt = pt_names[0]
# print(pt_names[0])

# # Filter slices by selected patient
# seg_slices = [slice_name for slice_name in seg_slices if pt in slice_name]
# expert_slices = [expert_name for expert_name in expert_slices if pt in expert_name]
# filenames = [image_name for image_name in filenames if pt in image_name]