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

def load_files(unet, dataset, subgroup):
    if(unet == "Fat"):
        seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/U_Net Code/Fat_Unet_Predictions_Excluded_" + subgroup + ".npy"
        expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/U_Net Code/Fat_Unet_Gt_Excluded_" + subgroup + ".npy"
        root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/Excluded/" + dataset + "/" + subgroup + "/"
    else:
        seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/U_Net Code/" + dataset + "_Unet_Gt_Excluded_" + subgroup + ".npy"
        expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/" + unet + " U-Net/U_Net Code/" + dataset + "_Unet_Predictions_Excluded_" + subgroup + ".npy"
        

        root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/Excluded/" + dataset + "/" + subgroup + "/"
        
    updated_subgroup = subgroup.replace("_"," ")    
    images1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Excluded Patients/Datasets/" + dataset + "_Testing_Dataset_excluded_" + updated_subgroup + ".hdf5"


    pt_names_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Excluded Patients/Subgroups/" + updated_subgroup + "/volumes/"
    
    # Get list of patient names
    pt_names = os.listdir(pt_names_path)
    pt_names = sorted([name.replace(".mha","") for name in pt_names])
    
    # Read in cropped images
    with h5py.File(images1_path, 'r') as f:
            images1 = f['testing_Images'][()]
            filenames1 = f['testing_image_filenames'][()]
            
    images1 = images1.squeeze()
    filenames1 = filenames1.tolist()
    filenames1=[x.decode('utf-8') for x in filenames1]
    
    seg1_slice_list_np = np.load(seg1_path)
    seg1_slice_list_np = seg1_slice_list_np.squeeze()

    expert1_slice_list_np = np.load(expert1_path)
    expert1_slice_list_np = expert1_slice_list_np.squeeze()
    
    return seg1_slice_list_np, expert1_slice_list_np, pt_names, images1, filenames1
  
# input args
subgroup = "dark_lumen"
pt_index = 0
slice_num = 22
    
orw_seg1_slice_list_np, orw_expert1_slice_list_np, orw_pt_names, orw_images1, orw_filenames1 = load_files("Outer Rectal Wall", "ORW", subgroup)
lumen_seg1_slice_list_np, lumen_expert1_slice_list_np, lumen_pt_names, lumen_images1, lumen_filenames1 = load_files("Lumen", "Lumen", subgroup)
fat_seg1_slice_list_np, fat_expert1_slice_list_np, fat_pt_names, fat_images1, fat_filenames1 = load_files("Fat", "Fat", subgroup)

orw_pt = orw_pt_names[pt_index]
lumen_pt = lumen_pt_names[pt_index]
fat_pt = fat_pt_names[pt_index]

assert orw_pt == lumen_pt == fat_pt

full_pt_name = orw_pt + "_" + str(slice_num)

assert orw_filenames1.index(full_pt_name) == lumen_filenames1.index(full_pt_name) == fat_filenames1.index(full_pt_name)

index_pos = orw_filenames1.index(full_pt_name)

# Specify Plot parameters

expert1_color = '#FF0068'
seg_color = '#00FF00'
thickness = 2
out_dpi = 300
a = 0.5

fig, ax = plt.subplots(3,1)

# # Plot Outer Rectal Wall
ax[0].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[0].set_title("ORW - " + subgroup)
ax[0].contour(orw_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[0].contour(orw_seg1_slice_list_np[index_pos], colors=seg_color, linewidths=thickness, alpha=a)
ax[0].grid(False)
ax[0].set_axis_off()

ax[1].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[1].set_title("Lumen - " + subgroup)
ax[1].contour(lumen_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[1].contour(lumen_seg1_slice_list_np[index_pos], colors=seg_color, linewidths=thickness, alpha=a)
ax[1].grid(False)
ax[1].set_axis_off()

ax[2].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[2].set_title("Fat - " + subgroup)
ax[2].contour(fat_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[2].contour(fat_seg1_slice_list_np[index_pos], colors=seg_color, linewidths=thickness, alpha=a)
ax[2].grid(False)
ax[2].set_axis_off()

fig.suptitle(full_pt_name)
