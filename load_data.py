#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:08:26 2021

@author: Tom
"""


import SimpleITK as sitk
import numpy as np
from resize_img import resize_img


def load_image(image_path):
    image = sitk.ReadImage(image_path)
    image_np = sitk.GetArrayFromImage(image)
    # Normalize images
    image_np = image_np - np.mean(image_np) #Mean normalization; substract the mean from each pixel value
    image_np = image_np / np.std(image_np) #Divide each pixel value by the standard deviation
    image_np = np.swapaxes(image_np, 0, 2)
    num_slices = image_np.shape[2]
    resized_slices = []
    for slice_num in range(num_slices):
        img_slice = image_np[:,:,slice_num] #Extract the image
        resized_slice = resize_img(img_slice, (128,128))
        resized_slice = resized_slice[:,:,None].repeat(1,axis=2)
        resized_slices.append(resized_slice)
    
    resized_imgs = np.array(resized_slices)
    return resized_imgs
    
    
