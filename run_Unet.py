#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:34:51 2021

@author: Tom
"""

import load_data as ld
import load_Unet as lm
import resize_img as ri
import numpy as np
import SimpleITK as sitk
import ntpath

# Load in the model
Model_Name = '/Volumes/GoogleDrive/Shared drives/INVent/tom/Post-CRT_Unet/Training_Lumen_Unet.hdf5'
unet = lm.load_Unet_model(Model_Name)

# Load in the image
image_path = '/Volumes/GoogleDrive/Shared drives/INVent_Data/Rectal/newdata/UH/RectalCA145-2/RectalCA145-2_Post_Ax.mha'
image_np, image_mha = ld.load_image(image_path)

# Predict on the slices
print ("Predicting on image...")
predictions = unet.predict(image_np, verbose=1)

# Threshold the slices
if('Lumen' in Model_Name):
    threshold = 0.901763019988
if('Outer_Rectal_Wall' in Model_Name):
    threshold = 0.976268057096
    
predictions = (predictions > threshold).astype(np.float32)

# Squeeze the predictions
predictions = predictions.squeeze()

# Reshape the predictions
predictions_reshaped = np.swapaxes(predictions, 0, 2) # Switch to (x,y,z)
new_size = image_mha.GetSize()
new_size = new_size[:2]
predictions_reshaped = ri.resize_img(predictions_reshaped, new_size, nn=True) # Resize to original image size
predictions_reshaped = np.transpose(predictions_reshaped,[2,0,1]) # Swtich to (z,y,x) because converting it to a .mha file will flip the axes back to (x,y,z)

# Set lumen label
predictions_reshaped[predictions_reshaped == 1] = 2

# Create export filename
filename = ntpath.basename(image_path)
export_name = filename[:len(filename)-4] + '_prediction_label' + filename[(len(filename)-4):]

# Export the mask as .mha file

export_vol = sitk.GetImageFromArray(predictions_reshaped)

export_vol.SetSpacing(image_mha.GetSpacing())
export_vol.SetDirection(image_mha.GetDirection())
export_vol.SetOrigin(image_mha.GetOrigin())

sitk.WriteImage(export_vol, export_name)

# test = sitk.PermuteAxesImageFilter()
# test.SetOrder([2,0,1])
# export_vol = test.Execute(export_vol)