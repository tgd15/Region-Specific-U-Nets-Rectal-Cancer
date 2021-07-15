#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:35:16 2021

@author: Tom
"""

import numpy as np
import scipy.io as sio
import os
#from mayavi import mlab
import skimage.measure as measure
import SimpleITK as sitk
import resize_img as ri
import h5py

def find_sub_list(sub_list,this_list):
    """
    https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list

    Parameters
    ----------
    sub_list : TYPE
        DESCRIPTION.
    this_list : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return (this_list.index(sub_list[0]),this_list.index(sub_list[-1]))

# Specify filepaths
seg_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Outer Rectal Wall U-Net/Results/CCA_Expert1/seg/"
expert_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Outer Rectal Wall U-Net/Results/ExperttoExpert/expert1/"
pt_names_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Volumes/"
images_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Datasets/ORW_Testing_Dataset_expert1.hdf5"

# Get list of patient names
pt_names = os.listdir(pt_names_path)
pt_names = sorted([name.replace(".mha","") for name in pt_names])

# Get list of slices
seg_slices = os.listdir(seg_path)
expert_slices = os.listdir(expert_path)

# Read in cropped images
with h5py.File(images_path, 'r') as f:
        all_images = f['testing_Images'][()]
        all_filenames = f['testing_image_filenames'][()]

all_images = all_images.squeeze()
all_filenames = all_filenames.tolist()
all_filenames=[x.decode('utf-8') for x in all_filenames]

# Select a patient
pt = pt_names[12]
print(pt)

# Filter slices by selected patient
seg_slices = [slice_name for slice_name in seg_slices if pt in slice_name]
expert_slices = [expert_name for expert_name in expert_slices if pt in expert_name]
filenames = [image_name for image_name in all_filenames if pt in image_name]

image_indices = find_sub_list(filenames, all_filenames)
images = all_images[image_indices[0]:image_indices[1]+1,:,:]

# Load in .mat files into array
seg_slice_list = []
expert_slice_list = []

# Load segmentation and expert slices into array

for i in range(len(seg_slices)):
    seg_slice = sio.loadmat(seg_path + seg_slices[i])
    seg_slice = seg_slice["prediction"]
    seg_slice_list.append(seg_slice)
    
    expert_slice = sio.loadmat(expert_path + expert_slices[i])
    expert_slice = expert_slice["actual"].squeeze()
    expert_slice_list.append(expert_slice)
    
# Convert lists to arrays
seg_slice_list_np = np.array(seg_slice_list)
expert_slice_list_np = np.array(expert_slice_list)

# Change expert label to 2
expert_slice_list_np[expert_slice_list_np == 1] = 2

# Load in the patient (reference) volume to initialize params for export
ref_vol = sitk.ReadImage(pt_names_path + pt +".mha")

new_size = ref_vol.GetSize()
new_size = new_size[:2]

# Create empty volume array
seg_vol = np.zeros(ref_vol.GetSize())
expert_vol = np.zeros(ref_vol.GetSize())

# Get slice numbers of expert and segmentation from filenames
slice_nums = [i.split('_',3)[3] for i in seg_slices]
slice_nums = [int(j.replace("_prediction.mat","")) for j in slice_nums]

# Switch axes from (z,y,x) to (x,y,z)
seg_slice_list_np = np.swapaxes(seg_slice_list_np, 0, 2) # Switch to (x,y,z)
expert_slice_list_np = np.swapaxes(expert_slice_list_np, 0, 2) # Switch to (x,y,z)
images = np.swapaxes(images, 0, 2)

# Resize the slices
seg_slice_list_np = ri.resize_img(seg_slice_list_np, new_size, nn=True) # Resize to original image size
expert_slice_list_np = ri.resize_img(expert_slice_list_np, new_size, nn=True) # Resize to original image size
images = ri.resize_img(images, new_size)

for k, index in enumerate(slice_nums):
    seg_vol[:,:,index] = seg_slice_list_np[:,:,k]
    expert_vol[:,:,index] = expert_slice_list_np[:,:,k]

# Undo the swap axes
seg_vol = np.swapaxes(seg_vol, 2, 0)
expert_vol = np.swapaxes(expert_vol, 2, 0)
images = np.swapaxes(images, 2, 0)
# seg_vol = np.transpose(seg_vol,[2,0,1]) # Swtich to (z,y,x) because converting it to a .mha file will flip the axes back to (x,y,z)
# expert_vol = np.transpose(expert_vol,[2,0,1]) # Swtich to (z,y,x) because converting it to a .mha file will flip the axes back to (x,y,z)
# images = np.transpose(images,[2,0,1])

# Create merge volume
merge_vol = seg_vol + expert_vol

# Create sitk Image
seg_vol = sitk.GetImageFromArray(seg_vol)
expert_vol = sitk.GetImageFromArray(expert_vol)
images_vol = sitk.GetImageFromArray(images)
merge_vol = sitk.GetImageFromArray(merge_vol)

# Copy export params from reference volume to segmentation and expert volumes
seg_vol.SetSpacing(ref_vol.GetSpacing())
seg_vol.SetDirection(ref_vol.GetDirection())
seg_vol.SetOrigin(ref_vol.GetOrigin())

expert_vol.SetSpacing(ref_vol.GetSpacing())
expert_vol.SetDirection(ref_vol.GetDirection())
expert_vol.SetOrigin(ref_vol.GetOrigin())

images_vol.SetSpacing(ref_vol.GetSpacing())
images_vol.SetDirection(ref_vol.GetDirection())
images_vol.SetOrigin(ref_vol.GetOrigin())

merge_vol.SetSpacing(ref_vol.GetSpacing())
merge_vol.SetDirection(ref_vol.GetDirection())
merge_vol.SetOrigin(ref_vol.GetOrigin())

# # Export segmentation and expert volumes as .mha
sitk.WriteImage(seg_vol, pt + "_Seg_Vol.mha")
sitk.WriteImage(expert_vol, pt + "_Expert_Vol.mha")
sitk.WriteImage(images_vol, pt + ".mha")
sitk.WriteImage(merge_vol, pt + "_merge_Vol.mha" )