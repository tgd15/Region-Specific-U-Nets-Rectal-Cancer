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

seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/Results/2022-02-24/CCA_VA/seg/"
expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/Results/2022-02-24/CCA_VA/gt/"
root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/VA/" + dataset + "/"
    
images1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/VA_Patients/Datasets/" + dataset + "_Testing_Dataset_VA.hdf5"
pt_names_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/VA_Patients/Volumes/"

# Get list of patient names
pt_names = os.listdir(pt_names_path)
pt_names = sorted([name.replace(".mha","") for name in pt_names])

# Get list of slices
seg1_slices = os.listdir(seg1_path)
expert1_slices = os.listdir(expert1_path)

# Read in cropped images
with h5py.File(images1_path, 'r') as f:
        images1 = f['testing_Images'][()]
        filenames1 = f['testing_image_filenames'][()]
        
images1 = images1.squeeze()
filenames1 = filenames1.tolist()
filenames1=[x.decode('utf-8') for x in filenames1]

# Read in data from .npy files
seg1_slice_list_np = np.load("/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Fat U-Net/U_Net Code/Fat_Unet_Predictions_VA.npy")
expert1_slice_list_np = np.load("/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Fat U-Net/U_Net Code/Fat_Unet_Gt_VA.npy")
expert1_slice_list_np = expert1_slice_list_np.squeeze()

# Specify Plot parameters

expert1_color = '#FF0068'
seg_color = '#00FF00'
thickness = 2
out_dpi = 300
a = 0.5

# Create plots
out_path = root_out + "Whole/"
for k in range(len(seg1_slices)):
    plt.axis('off')
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    
    ax[0].imshow(images1[k], cmap='gray', interpolation='bilinear')
    # ax[0].set_title('Image')
    ax[0].grid(False)
    
    ax[1].imshow(images1[k], cmap='gray', interpolation='bilinear')
    ax[1].contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
    # ax[1].set_title('Image Annotated Expert')
    ax[1].grid(False)
    
    ax[2].imshow(images1[k], cmap='gray', interpolation='bilinear')
    ax[2].contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
    ax[2].contour(seg1_slice_list_np[k], colors='#00FF00', linewidths=thickness, alpha=a)
    # ax[2].set_title('U-Net Compared to Expert')
    ax[2].grid(False)
    
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=400)
    plt.close()
    
# Create Original Image plots
out_path = root_out + "Original_Image/"
for k in range(len(seg1_slices)):
    plt.axis('off')
    fig = plt.figure(frameon=False)
    plt.imshow(images1[k], cmap='gray', interpolation='bilinear')
    # plt.title('Image')
    plt.grid(False)
    plt.axis("off")
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=400)
    plt.close()
    
# Create image with expert annotation
out_path = root_out + "Expert/"
for k in range(len(seg1_slices)):
    plt.axis('off')
    fig = plt.figure(frameon=False)
    plt.imshow(images1[k], cmap='gray', interpolation='bilinear')
    plt.contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
    # plt.title('Image with Expert Annotation')
    plt.grid(False)
    plt.axis("off")
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=400)
    plt.close()
    
# Create image with segmentation and expert annotation
out_path = root_out + "Segmentation/"
for k in range(len(seg1_slices)):
    plt.axis('off')
    fig = plt.figure(frameon=False)
    plt.imshow(images1[k], cmap='gray', interpolation='bilinear')
    plt.contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
    plt.contour(seg1_slice_list_np[k], colors='#00FF00', linewidths=thickness, alpha=a)
    # plt.title('Image with Expert Annotation')
    plt.grid(False)
    plt.axis("off")
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=400)
    plt.close()
    
    
# Select a patient
# pt = pt_names[0]
# print(pt_names[0])

# # Filter slices by selected patient
# seg_slices = [slice_name for slice_name in seg_slices if pt in slice_name]
# expert_slices = [expert_name for expert_name in expert_slices if pt in expert_name]
# filenames = [image_name for image_name in filenames if pt in image_name]