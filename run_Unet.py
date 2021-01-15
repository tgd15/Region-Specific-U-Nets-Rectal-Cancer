#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:34:51 2021

@author: Tom
"""

import load_model as lm

Model_Name = '/Volumes/GoogleDrive/Shared drives/INVent/tom/Post-CRT_Unet/Training_Lumen_Unet.hdf5'

print ("Predicting on original masks...")
holdoutTest = '/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-Revised/Testing/Datasets/Lumen_Testing_Dataset_original_masks.hdf5'
predictionName = 'Lumen_Preds_Original_Masks.hdf5'
lm.load_and_predict(Model_Name,holdoutTest,predictionName)
print ("Saved predictions \n")

print ("Predicting on updated masks...")
holdoutTest = '/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-Revised/Testing/Datasets/Lumen_Testing_Dataset_updated_masks.hdf5'
predictionName = 'Lumen_Preds_Updated_Masks.hdf5'
lm.load_and_predict(Model_Name,holdoutTest,predictionName)
print ("Saved predictions")