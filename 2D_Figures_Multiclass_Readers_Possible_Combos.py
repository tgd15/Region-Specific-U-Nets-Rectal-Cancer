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
        seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Predictions_expert1.npy"
        seg2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Predictions_expert2.npy"
        expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Gt_expert1.npy"
        expert2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Gt_expert2.npy"
        root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/Multiple_Experts/" + unet + "/"
    else:
        seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Predictions_expert1.npy"
        seg2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Predictions_expert2.npy"
        expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Gt_expert1.npy"
        expert2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Gt_expert2.npy"
        root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/Multiple_Experts/" + dataset+ "/"
        
    images1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Datasets/" + dataset + "_Testing_Dataset_expert1.hdf5"
    images2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Datasets/" + dataset + "_Testing_Dataset_expert2.hdf5"

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

    # Read .npy files

    seg1_slice_list_np = np.load(seg1_path)
    seg2_slice_list_np = np.load(seg2_path)

    expert1_slice_list_np = np.load(expert1_path)
    expert2_slice_list_np = np.load(expert2_path)

    expert1_slice_list_np = expert1_slice_list_np.squeeze()
    expert2_slice_list_np = expert2_slice_list_np.squeeze()

    return images1, filenames1, images2, filenames2, seg1_slice_list_np, seg2_slice_list_np, expert1_slice_list_np, expert2_slice_list_np, root_out


orw_images1, orw_filenames1, orw_images2, orw_filenames2, orw_seg1_slice_list_np, orw_seg2_slice_list_np, orw_expert1_slice_list_np, orw_expert2_slice_list_np, orw_root_out = load_files("Outer Rectal Wall", "ORW")
lumen_images1, lumen_filenames1, lumen_images2, lumen_filenames2, lumen_seg1_slice_list_np, lumen_seg2_slice_list_np, lumen_expert1_slice_list_np, lumen_expert2_slice_list_np, lumen_root_out = load_files("Lumen", "Lumen")
fat_images1, fat_filenames1, fat_images2, fat_filenames2, fat_seg1_slice_list_np, fat_seg2_slice_list_np, fat_expert1_slice_list_np, fat_expert2_slice_list_np, fat_root_out = load_files("Fat", "Fat")

# Specify filepaths
# unet = input("Unet Name: ")
# dataset = input("Dataset Name: ")


pt_names_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Volumes/"

# Get list of patient names

pt_names = os.listdir(pt_names_path)
pt_names = sorted([name.replace(".mha","") for name in pt_names])
try:
    pt_names.remove(".DS_Store")
except Exception as e:
    print(".DS_Store not present in patinet list. yay!")
    

pt = pt_names[18]
slice_num = 18

full_pt_name = pt + "_" + str(slice_num)
index_pos = orw_filenames1.index(full_pt_name)


# Specify Plot parameters

expert1_color = '#FF0068'
expert2_color = '#00C5FF'
seg_color = '#00FF00'
thickness = 2
out_dpi = 400
a = 0.5

fig, ax = plt.subplots(3, 4, figsize = (20, 10))

# Plot Outer Rectal Wall
ax[0, 0].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[0, 0].set_title('Image')
ax[0, 0].grid(False)
ax[0, 0].set_axis_off()

ax[0, 1].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[0, 1].contour(orw_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[0, 1].contour(orw_expert2_slice_list_np[index_pos], colors=expert2_color, linewidths=thickness, alpha=a)
ax[0, 1].set_title('Image Annotated by 2 experts \n (Red = Expert 1 and Cyan = Expert 2)')
ax[0, 1].grid(False)
ax[0, 1].set_axis_off()

ax[0, 2].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[0, 2].contour(orw_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[0, 2].contour(orw_seg1_slice_list_np[index_pos], colors=seg_color, linewidths=thickness, alpha=a)
ax[0, 2].set_title('U-Net Compared to Expert 1')
ax[0, 2].grid(False)
ax[0, 2].set_axis_off()

ax[0, 3].imshow(orw_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[0, 3].contour(orw_expert2_slice_list_np[index_pos], colors=expert2_color, linewidths=thickness, alpha=a)
ax[0, 3].contour(orw_seg2_slice_list_np[index_pos], colors=seg_color, linewidths=thickness, alpha=a)
ax[0, 3].set_title('U-Net Compared to Expert 2')
ax[0, 3].grid(False)
ax[0, 3].set_axis_off()

# Plot Lumen
ax[1, 0].imshow(lumen_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[1, 0].set_title('Image')
ax[1, 0].grid(False)
ax[1, 0].set_axis_off()

ax[1, 1].imshow(lumen_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[1, 1].contour(lumen_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[1, 1].contour(lumen_expert2_slice_list_np[index_pos], colors=expert2_color, linewidths=thickness, alpha=a)
ax[1, 1].set_title('Image Annotated by 2 experts \n (Red = Expert 1 and Cyan = Expert 2)')
ax[1, 1].grid(False)
ax[1, 1].set_axis_off()

ax[1, 2].imshow(lumen_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[1, 2].contour(lumen_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[1, 2].contour(lumen_seg1_slice_list_np[index_pos], colors=seg_color, linewidths=thickness, alpha=a)
ax[1, 2].set_title('U-Net Compared to Expert 1')
ax[1, 2].grid(False)
ax[1, 2].set_axis_off()

ax[1, 3].imshow(lumen_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[1, 3].contour(lumen_expert2_slice_list_np[index_pos], colors=expert2_color, linewidths=thickness, alpha=a)
ax[1, 3].contour(lumen_seg2_slice_list_np[index_pos], colors=seg_color, linewidths=thickness, alpha=a)
ax[1, 3].set_title('U-Net Compared to Expert 2')
ax[1, 3].grid(False)
ax[1, 3].set_axis_off()

# Plot Fat
ax[2, 0].imshow(fat_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[2, 0].set_title('Image')
ax[2, 0].grid(False)
ax[2, 0].set_axis_off()

ax[2, 1].imshow(fat_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[2, 1].contour(fat_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[2, 1].contour(fat_expert2_slice_list_np[index_pos], colors=expert2_color, linewidths=thickness, alpha=a)
ax[2, 1].set_title('Image Annotated by 2 experts \n (Red = Expert 1 and Cyan = Expert 2)')
ax[2, 1].grid(False)
ax[2, 1].set_axis_off()

ax[2, 2].imshow(fat_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[2, 2].contour(fat_expert1_slice_list_np[index_pos], colors=expert1_color, linewidths=thickness, alpha=a)
ax[2, 2].contour(fat_seg1_slice_list_np[index_pos], colors=seg_color, linewidths=thickness, alpha=a)
ax[2, 2].set_title('U-Net Compared to Expert 1')
ax[2, 2].grid(False)
ax[2, 2].set_axis_off()

ax[2, 3].imshow(fat_images1[index_pos], cmap='gray', interpolation='bilinear')
ax[2, 3].contour(fat_expert2_slice_list_np[index_pos], colors=expert2_color, linewidths=thickness, alpha=a)
ax[2, 3].contour(fat_seg2_slice_list_np[index_pos], colors=seg_color, linewidths=thickness, alpha=a)
ax[2, 3].set_title('U-Net Compared to Expert 2')
ax[2, 3].grid(False)
ax[2, 3].set_axis_off()

fig.suptitle(full_pt_name)