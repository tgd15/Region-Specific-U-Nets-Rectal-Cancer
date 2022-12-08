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

def load_files(unet, dataset):

    if(unet == "Fat"):
        seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/U_Net Code/" + unet + "_Unet_Predictions_VA.npy"
        expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/U_Net Code/" + unet + "_Unet_Gt_VA.npy"
        root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/VA/" + dataset + "/"
    else:
        seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/U_Net Code/" + dataset + "_Unet_Predictions_VA.npy"
        expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/U_Net Code/" + dataset + "_Unet_Gt_VA.npy"
        root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/VA/" + dataset + "/"
        
    images1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/VA_Patients/Datasets/" + dataset + "_Testing_Dataset_VA.hdf5"
    
    # Read in cropped images
    with h5py.File(images1_path, 'r') as f:
            images1 = f['testing_Images'][()]
            filenames1 = f['testing_image_filenames'][()]
            
    images1 = images1.squeeze()
    filenames1 = filenames1.tolist()
    filenames1=[x.decode('utf-8') for x in filenames1]
    
    # Read in data from .npy files
    seg1_slice_list_np = np.load(seg1_path)
    expert1_slice_list_np = np.load(expert1_path)
    expert1_slice_list_np = expert1_slice_list_np.squeeze()
    
    return seg1_slice_list_np, expert1_slice_list_np, images1, filenames1
    
# Pt 11, 23 and Pt 5 , 29
pt_index = 3
slice_num = 29

orw_seg1_slice_list_np, orw_expert1_slice_list_np, orw_images1, orw_filenames1 = load_files("Outer Rectal Wall", "ORW")
lumen_seg1_slice_list_np, lumen_expert1_slice_list_np, lumen_images1, lumen_filenames1 = load_files("Lumen", "Lumen")
fat_seg1_slice_list_np, fat_expert1_slice_list_np, fat_images1, fat_filenames1 = load_files("Fat", "Fat")

    
pt_names_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/VA_Patients/Volumes/"

# Get list of patient names
pt_names = os.listdir(pt_names_path)
pt_names = sorted([name.replace(".mha","") for name in pt_names])

pt_name = pt_names[pt_index]

full_pt_name = pt_name + "_" + str(slice_num)

assert orw_filenames1.index(full_pt_name) == lumen_filenames1.index(full_pt_name) == fat_filenames1.index(full_pt_name)

index_pos = orw_filenames1.index(full_pt_name)



# Specify Plot parameters

expert1_color = '#FF0068'
seg_color = '#00FF00'
thickness = 2
out_dpi = 300
a = 0.5

# Create plots
plt.axis('off')
fig, ax = plt.subplots(3, 3, figsize=(20, 10))

# Outer Rectal Wall
ax[0, 0].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[0, 0].set_title('Image')
ax[0, 0].grid(False)
ax[0, 0].set_axis_off()

ax[0, 1].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[0, 1].contour(orw_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[0, 1].set_title('Image Annotated Expert')
ax[0, 1].grid(False)
ax[0, 1].set_axis_off()

ax[0, 2].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[0, 2].contour(orw_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[0, 2].contour(orw_seg1_slice_list_np[index_pos], colors='#00FF00', linewidths=thickness, alpha=a)
ax[0, 2].set_title('U-Net Compared to Expert')
ax[0, 2].grid(False)
ax[0, 2].set_axis_off()

# Lumen
ax[1, 0].imshow(lumen_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[1, 0].set_title('Image')
ax[1, 0].grid(False)
ax[1, 0].set_axis_off()

ax[1, 1].imshow(lumen_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[1, 1].contour(lumen_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[1, 1].set_title('Image Annotated Expert')
ax[1, 1].grid(False)
ax[1, 1].set_axis_off()

ax[1, 2].imshow(lumen_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[1, 2].contour(lumen_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[1, 2].contour(lumen_seg1_slice_list_np[index_pos], colors='#00FF00', linewidths=thickness, alpha=a)
ax[1, 2].set_title('U-Net Compared to Expert')
ax[1, 2].grid(False)
ax[1, 2].set_axis_off()

# Fat
ax[2, 0].imshow(fat_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[2, 0].set_title('Image')
ax[2, 0].grid(False)
ax[2, 0].set_axis_off()

ax[2, 1].imshow(fat_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[2, 1].contour(fat_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[2, 1].set_title('Image Annotated Expert')
ax[2, 1].grid(False)
ax[2, 1].set_axis_off()

ax[2, 2].imshow(fat_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[2, 2].contour(fat_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[2, 2].contour(fat_seg1_slice_list_np[index_pos], colors='#00FF00', linewidths=thickness, alpha=a)
ax[2, 2].set_title('U-Net Compared to Expert')
ax[2, 2].grid(False)
ax[2, 2].set_axis_off()


fig.suptitle(full_pt_name)



