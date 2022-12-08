#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:02:12 2022

@author: Tom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_excel(excel_path):
    
    xls = pd.ExcelFile(excel_path)
    xls_dict = {}
    for sheet_name in xls.sheet_names:
        xls_dict[sheet_name] = xls.parse(sheet_name)
        
    return xls_dict

subgroup = "dark_lumen"

# Read in DSC scores
orw_expert1_path = "../Outer Rectal Wall U-Net/Results/CCA_Excluded_" + subgroup + "/Dice_Per_Slice.xlsx"
lumen_expert1_path = "../Lumen U-Net/Results/CCA_Excluded_" + subgroup + "/Dice_Per_Slice.xlsx"
fat_expert1_path = "../Fat U-Net/Results/2022-02-24/CCA_Excluded_" + subgroup + "/Dice_Per_Slice.xlsx"


orw_expert1_dsc = parse_excel(orw_expert1_path)
lumen_expert1_dsc = parse_excel(lumen_expert1_path)
fat_expert1_dsc = parse_excel(fat_expert1_path)

# Get list of patients
subgroup =subgroup.replace("_", " ")
pt_dir = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Excluded Patients/Subgroups/" + subgroup + "/volumes/"
pt_files = sorted([os.path.splitext(filename)[0] for filename in os.listdir(pt_dir)])

# Loop through each patient and plot DSC scores
i = 0
for patient in pt_files:
    
    # Get DSC scores
    orw_expert1 = orw_expert1_dsc[patient]
    lumen_expert1 = lumen_expert1_dsc[patient]
    fat_expert1 = fat_expert1_dsc[patient]
 
    # Initialize plot
    fig, ax = plt.subplots(sharey = True, figsize = (10, 10))
    
    # PLOT 1
    # Plot lumen DSC
    x = list(lumen_expert1.diceInfo_1)
    x = [ele.split("_")[-1] for ele in x]
    y = lumen_expert1.diceInfo_2
    
    ax.plot(x, y, label = "Lumen Expert 1", marker = "o", color = "saddlebrown")
    
    # Plot ORW DSC
    x = list(orw_expert1.diceInfo_1)
    y = orw_expert1.diceInfo_2
    x = [ele.split("_")[-1] for ele in x]
    ax.plot(x, y, label = "ORW Expert 1", marker = "o", color = "blue")
    
    # Plot fat DSC
    # Some slice of lumen/ORW have no fat.
    # This will cause a mismatch of x andy dimensions on plot
    # Alter fat DSC dataframe to have same number of rows as ORW dataframe
    if(len(x) != fat_expert1.diceInfo_1.shape[0]):
        difference = orw_expert1.loc[~orw_expert1['diceInfo_1'].isin(fat_expert1['diceInfo_1'])]
        difference.loc[difference['diceInfo_2'] > 0.0, "diceInfo_2"] = 0
        
        fat_expert1 = fat_expert1.append(difference, ignore_index = True)
        fat_expert1 = fat_expert1.sort_values(by = ["diceInfo_1"])

    y = fat_expert1.diceInfo_2
    
    ax.plot(x, y, label = "Fat Expert 1", marker = "o", color = "green")
    ax.set_xticklabels(x)
    ax.yaxis.set_ticks(np.arange(0, 1, 0.1))
    ax.set_xlabel("Slice Number")
    ax.set_ylabel("DSC")
    ax.set_title(patient)
    
    fig.savefig("2D Figures/Comparisons/Excluded/" + patient + ".png", dpi = 300, bbox_inches = "tight")

    