#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:34:51 2021

@author: Tom
"""

import load_data as ld
import load_Unet as lm

Model_Name = '/Volumes/GoogleDrive/Shared drives/INVent/tom/Post-CRT_Unet/Training_Lumen_Unet.hdf5'

image_path = '/Volumes/GoogleDrive/Shared drives/INVent_Data/Rectal/newdata/UH/RectalCA130-2/RectalCA130-2_Post_Ax.mha'
image = ld.load_image(image_path)
unet = lm.load_Unet_model(Model_Name)

print ("Predicting on image...")
predictions = unet.predict(image, verbose=1)