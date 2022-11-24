#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:43:06 2022

@author: Tom
"""

import matplotlib.pyplot as plt
import os
from PIL import Image

site = input("Which site? ")
pt_id = input("Which patient? ")
slice_num = input("What slice number? ")

orw_path = "2D Figures/Multiple_Experts/ORW/Whole/"
orw_files = os.listdir(orw_path)

lumen_path = "2D Figures/Multiple_Experts/Lumen/Whole/"
lumen_files = os.listdir(lumen_path)

fat_path = "2D Figures/Multiple_Experts/Fat/Whole/"
fat_files = os.listdir("2D Figures/Multiple_Experts/Fat/Whole/")

patient_file = site + "_RectalCA-" + pt_id + "_" + slice_num + ".png"

orw_img = Image.open(orw_path + patient_file)
lumen_img = Image.open(lumen_path + patient_file)
fat_img = Image.open(fat_path + patient_file)

fig, ax = plt.subplots(3, 1)

ax[0].imshow(orw_img)
ax[1].imshow(lumen_img)
ax[2].imshow(fat_img)


