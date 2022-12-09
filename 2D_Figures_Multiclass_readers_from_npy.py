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
dataset = input("Dataset Name: ")

# if(unet == "Fat"):
#         seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Predictions_expert1.npy"
#         seg2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Predictions_expert2.npy"
#         expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Gt_expert1.npy"
#         expert2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Gt_expert2.npy"
#         root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/Multiple_Experts/" + unet + "/"
# else:
seg1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Predictions_expert1.npy"
seg2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Predictions_expert2.npy"
expert1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Gt_expert1.npy"
expert2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Multiclass U-Net/U_Net Code/npy_files/" + dataset + "/" + dataset + "_Multiclass_Unet_Gt_expert2.npy"
root_out = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/2D Figures/Multiclass/Multiple_Experts/" + dataset + "/"
    
images1_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Datasets/" + dataset + "_Testing_Dataset_expert1.hdf5"
images2_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Datasets/" + dataset + "_Testing_Dataset_expert2.hdf5"
pt_names_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Volumes/"

# Get list of patient names

pt_names = os.listdir(pt_names_path)
pt_names = sorted([name.replace(".mha","") for name in pt_names])
try:
    pt_names.remove(".DS_Store")
except Exception as e:
    print(".DS_Store not present in patinet list. yay!")

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


# Specify Plot parameters

expert1_color = '#FF0068'
expert2_color = '#00C5FF'
seg_color = '#00FF00'
thickness = 2
out_dpi = 400
a = 0.5

# Create plots
out_path = root_out + "Whole/"
for k in range(len(seg1_slice_list_np)):
    plt.axis('off')
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    
    ax[0].imshow(images1[k], cmap='gray', interpolation='bilinear')
    #ax[0].set_title('Image')
    ax[0].grid(False)
    
    ax[1].imshow(images1[k], cmap='gray', interpolation='bilinear')
    ax[1].contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
    ax[1].contour(expert2_slice_list_np[k], colors=expert2_color, linewidths=thickness, alpha=a)
    #ax[1].set_title('Image Annotated by 2 experts \n (Red = Expert 1 and Cyan = Expert 2)')
    ax[1].grid(False)
    
    ax[2].imshow(images1[k], cmap='gray', interpolation='bilinear')
    ax[2].contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
    ax[2].contour(seg1_slice_list_np[k], colors=seg_color, linewidths=thickness, alpha=a)
    #ax[2].set_title('U-Net Compared to Expert 1')
    ax[2].grid(False)
    
    ax[3].imshow(images1[k], cmap='gray', interpolation='bilinear')
    ax[3].contour(expert2_slice_list_np[k], colors=expert2_color, linewidths=thickness, alpha=a)
    ax[3].contour(seg2_slice_list_np[k], colors=seg_color, linewidths=thickness, alpha=a)
    #ax[3].set_title('U-Net Compared to Expert 2')
    ax[3].grid(False)
    
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    ax[3].set_axis_off()
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=out_dpi)
    plt.close()
    
# Create Original Image plots
out_path = root_out + "Original/"
for k in range(len(seg1_slice_list_np)):
    plt.axis('off')
    fig = plt.figure(frameon=False)
    plt.imshow(images1[k], cmap='gray', interpolation='bilinear')
    #plt.title('Image')
    plt.grid(False)
    plt.axis("off")
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=out_dpi)
    plt.close()
    
# Create image with expert annotation
out_path = root_out + "Expert/"
for k in range(len(seg1_slice_list_np)):
    plt.axis('off')
    fig = plt.figure(frameon=False)
    plt.imshow(images1[k], cmap='gray', interpolation='bilinear')
    plt.contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
    plt.contour(expert2_slice_list_np[k], colors=expert2_color, linewidths=thickness, alpha=a)
    #plt.title('Image with Expert Annotation')
    plt.grid(False)
    plt.axis("off")
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=out_dpi)
    fig.canvas.flush_events()
    plt.close()
    
# Create image with segmentation and expert annotation 1
out_path = root_out + "Segmentation-Expert1/"
for k in range(len(seg1_slice_list_np)):
    plt.axis('off')
    fig = plt.figure(frameon=False)
    plt.imshow(images1[k], cmap='gray', interpolation='bilinear')
    plt.contour(expert1_slice_list_np[k], colors=expert1_color, linewidths=thickness, alpha=a)
    plt.contour(seg1_slice_list_np[k], colors=seg_color, linewidths=thickness, alpha=a)
    #plt.title('Image with Expert Annotation')
    plt.grid(False)
    plt.axis("off")
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=out_dpi)
    plt.close()
    
# Create image with segmentation and expert annotation 2
out_path = root_out + "Segmentation-Expert2/"
for k in range(len(seg1_slice_list_np)):
    plt.axis('off')
    fig = plt.figure(frameon=False)
    plt.imshow(images1[k], cmap='gray', interpolation='bilinear')
    plt.contour(expert2_slice_list_np[k], colors=expert2_color, linewidths=thickness, alpha=a)
    plt.contour(seg2_slice_list_np[k], colors=seg_color, linewidths=thickness, alpha=a)
    #plt.title('Image with Expert Annotation')
    plt.grid(False)
    plt.axis("off")
    fig.savefig(out_path + filenames1[k] + '.png', bbox_inches='tight', dpi=out_dpi)
    plt.close()