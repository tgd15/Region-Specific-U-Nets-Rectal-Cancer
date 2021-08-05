#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:35:16 2021

@author: Tom
"""

import numpy as np
import scipy.io as sio
import os
from pathlib import Path
import SimpleITK as sitk
import h5py

def parse_path(expertpath, segpath):
    """Parse filepath to excel document containing metrics.
    
    Parse a filepath to an excel document containing metrics to get the following information:
        - U-Net type
        - Expert Number (Expert 1 or Expert 2)
        - Metric (Dice, Hausdorff, FD)
    

    Parameters
    ----------
    expertpath : str
        Path to excel document containing metrics for expert 1 vs. expert 2.
    segpath : str
        Path to excel document containing metrics for U-Net vs. Expert.

    Raises
    ------
    ValueError
        Raise error when expertpath and segpath do not have the same U-Net and metric in filename.

    Returns
    -------
    unet : str
        U-Net name.
    expert : str
        Expert 1 or Expert 2.
    metric : str
        Metric name.
    """
    
    def do_parse(path):
        """Sub-function for actually parsing filename.
        
        Parameters
        ----------
        path : str
            Filename to parse.

        Returns
        -------
        unet : str
            U-Net name.
        expert : str
            Expert 1 or Expert 2.
        metric : str
            Metric name.
        """
        thepath = Path(path)
        unet = thepath.parts[7]
        return unet
    
    ex_unet = do_parse(expertpath)
    seg_unet = do_parse(segpath)
    
    if(ex_unet == seg_unet):
        unet = ex_unet
    elif(ex_unet == "Rectal Wall U-Net" and seg_unet == "Segmentation Fusion"):
        unet = seg_unet
    else:
        raise ValueError("Filepaths do not correspond to each other!")
    
    return unet

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

# Ask for expert
expert = input("Please specify 'expert1' or 'expert2':")
patient = input("Please specify a patient ID: ")

# Capitalize first letter for the seg_path
expert_for_seg_path = expert.capitalize()
    
# Specify filepaths
seg_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Segmentation Fusion/Results/" + expert_for_seg_path + "/seg/"
expert_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Rectal Wall U-Net/Results/ExperttoExpert/"+ expert + "/"
pt_names_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Volumes/"
images_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Datasets/Rectal_Wall_Testing_Dataset_"+ expert + ".hdf5"

unet = parse_path(expert_path, seg_path)

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

# Create directories
#pt = pt_names[12]
index = pt_names.index(patient)
pt = pt_names[index]
print("Writing volumes for " + pt)
if(os.path.isdir(pt) is False):
    os.mkdir(pt)
   
base_output = pt + "/" + unet
if(os.path.isdir(base_output) is False):
    os.mkdir(base_output)

# Create output filenames
seg_vol_name = base_output + "/" + pt + "_Seg_Vol_for_" + expert + ".mha"
expert_vol_name = base_output + "/" + pt + "_" + expert + "_Vol_gt.mha"
image_vol_name = base_output + "/" + pt + "_Image_Vol.mha"
merge_vol_name = base_output + "/" + pt + "_merge_Vol_for_" + expert + ".mha"

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

#new_size = ref_vol.GetSize()
new_size = images.shape
new_size = new_size[:2]

# Create empty volume array
seg_vol = np.zeros(images.shape)
seg_vol = np.swapaxes(seg_vol, 0, 2)
expert_vol = np.zeros(images.shape)
expert_vol = np.swapaxes(expert_vol, 0, 2)

# Get slice numbers of expert and segmentation from filenames
slice_nums = [i.split('_',3)[3] for i in seg_slices]
slice_nums = [int(j.replace("_prediction.mat","")) for j in slice_nums]

# Switch axes from (z,y,x) to (x,y,z)
seg_slice_list_np = np.swapaxes(seg_slice_list_np, 0, 2) # Switch to (x,y,z)
expert_slice_list_np = np.swapaxes(expert_slice_list_np, 0, 2) # Switch to (x,y,z)
images = np.swapaxes(images, 0, 2)

for k in range(len(slice_nums)):
    seg_vol[:,:,k] = seg_slice_list_np[:,:,k]
    expert_vol[:,:,k] = expert_slice_list_np[:,:,k]

# Undo the swap axes
seg_vol = np.swapaxes(seg_vol, 2, 0)
expert_vol = np.swapaxes(expert_vol, 2, 0)
images = np.swapaxes(images, 2, 0)

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

# Export segmentation and expert volumes as .mha
sitk.WriteImage(seg_vol, seg_vol_name)
sitk.WriteImage(expert_vol, expert_vol_name)
sitk.WriteImage(images_vol, image_vol_name)
#sitk.WriteImage(merge_vol, merge_vol_name)

print("\n All volumes have been generated for " + pt + ". You can load them into Paraview")