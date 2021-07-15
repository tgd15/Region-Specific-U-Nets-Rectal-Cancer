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
subgroup = input("Subgroup: ")

seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/Results/CCA_Excluded_" + subgroup + "/seg/"
expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/Results/CCA_Excluded_" + subgroup + "/gt/"
root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/Excluded/" + dataset + "/" + subgroup + "/"
    
updated_subgroup = subgroup.replace("_"," ")    
images1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Excluded Patients/Datasets/" + dataset + "_Testing_Dataset_excluded_" + updated_subgroup + ".hdf5"


pt_names_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Excluded Patients/Subgroups/" + updated_subgroup + "/"

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


# Load in .mat files into array
seg1_slice_list = []
expert1_slice_list = []

# Load segmentation and expert slices into array

for i in range(len(seg1_slices)):
    seg1_slice = sio.loadmat(seg1_path + seg1_slices[i])
    seg1_slice = seg1_slice["prediction"]
    seg1_slice_list.append(seg1_slice)
    
    expert1_slice = sio.loadmat(expert1_path + expert1_slices[i])
    expert1_slice = expert1_slice["actual"].squeeze()
    expert1_slice_list.append(expert1_slice)
    
# Convert lists to arrays
seg1_slice_list_np = np.array(seg1_slice_list)
expert1_slice_list_np = np.array(expert1_slice_list)

# Specify Plot parameters

expert1_color = '#FF0068'
seg_color = '#00FF00'
thickness = 2
out_dpi = 300
a = 0.5

# # Create whole plots
# out_path = root_out + "Whole/"
# for k in range(len(seg1_slices)):
#     plt.axis('off')
#     fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    
#     ax[0].imshow(images1[k], cmap='gray', interpolation='bilinear')
#     # ax[0].set_title('Image')
#     ax[0].grid(False)
    
#     ax[1].imshow(images1[k], cmap='gray', interpolation='bilinear')
#     ax[1].contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
#     # ax[1].set_title('Image Annotated by Expert')
#     ax[1].grid(False)
    
#     ax[2].imshow(images1[k], cmap='gray', interpolation='bilinear')
#     ax[2].contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
#     ax[2].contour(seg1_slice_list_np[k], colors='#00FF00', linewidths=thickness, alpha=a)
#     # ax[2].set_title('U-Net Compared to Expert 1')
#     ax[2].grid(False)
    
#     ax[0].set_axis_off()
#     ax[1].set_axis_off()
#     ax[2].set_axis_off()
#     fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=out_dpi)
#     plt.close()
    

# # Create Original Image plots
# out_path = root_out + "Original_Image/"
# for k in range(len(seg1_slices)):
#     plt.axis('off')
#     fig = plt.figure(frameon=False)
#     plt.imshow(images1[k], cmap='gray', interpolation='bilinear')
#     # plt.title('Image')
#     plt.grid(False)
#     plt.axis("off")
#     fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=out_dpi)
#     plt.close()
    
# # Create image with expert annotation
# out_path = root_out + "Expert/"
# for k in range(len(seg1_slices)):
#     plt.axis('off')
#     fig = plt.figure(frameon=False)
#     plt.imshow(images1[k], cmap='gray', interpolation='bilinear')
#     plt.contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
#     # plt.title('Image with Expert Annotation')
#     plt.grid(False)
#     plt.axis("off")
#     fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=out_dpi)
#     plt.close()
    
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
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=out_dpi)
    plt.close()