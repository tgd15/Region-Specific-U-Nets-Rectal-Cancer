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

# Read in ORW DSC scores
orw_expert1_path = "../Outer Rectal Wall U-Net/Results/CCA_Expert1/Dice_Per_Slice.xlsx"
orw_expert2_path = "../Outer Rectal Wall U-Net/Results/CCA_Expert2/Dice_Per_Slice.xlsx"
orw_reader_path = "../Outer Rectal Wall U-Net/Results/ExperttoExpert/Dice_Per_Slice.xlsx"

orw_expert1_dsc = parse_excel(orw_expert1_path)
orw_expert2_dsc = parse_excel(orw_expert2_path)
orw_reader_dsc = parse_excel(orw_reader_path)

# Read in Lumen DSC scores
lumen_expert1_path = "../Lumen U-Net/Results/CCA_Expert1/Dice_Per_Slice.xlsx"
lumen_expert2_path = "../Lumen U-Net/Results/CCA_Expert2/Dice_Per_Slice.xlsx"
lumen_reader_path = "../Lumen U-Net/Results/ExperttoExpert/Dice_Per_Slice.xlsx"

lumen_expert1_dsc = parse_excel(lumen_expert1_path)
lumen_expert2_dsc = parse_excel(lumen_expert2_path)
lumen_reader_dsc = parse_excel(lumen_reader_path)

# Read in Fat DSC scores
fat_expert1_path = "../Fat U-Net/Results/2022-02-24/CCA_Expert1/Dice_Per_Slice.xlsx"
fat_expert2_path = "../Fat U-Net/Results/2022-02-24/CCA_Expert2/Dice_Per_Slice.xlsx"
fat_reader_path = "../Fat U-Net/Results/2022-02-24/ExperttoExpert/Dice_Per_Slice.xlsx"

fat_expert1_dsc = parse_excel(fat_expert1_path)
fat_expert2_dsc = parse_excel(fat_expert2_path)
fat_reader_dsc = parse_excel(fat_reader_path)

# Get list of patients
pt_dir = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Testing/Volumes/"
pt_files = sorted([os.path.splitext(filename)[0] for filename in os.listdir(pt_dir)])

# Loop through each patient and plot DSC scores
i = 0
for patient in pt_files:
    
    # Get DSC scores
    orw_expert1 = orw_expert1_dsc[patient]
    lumen_expert1 = lumen_expert1_dsc[patient]
    fat_expert1 = fat_expert1_dsc[patient]

    orw_expert2 = orw_expert2_dsc[patient]
    lumen_expert2 = lumen_expert2_dsc[patient]
    fat_expert2 = fat_expert2_dsc[patient]
 
    # Initialize plot
    fig, ax = plt.subplots(2, 1, sharey = True, figsize = (10, 10))
    
    # PLOT 1
    # Plot lumen DSC
    x = list(lumen_expert1.diceInfo_1)
    x = [ele.split("_")[-1] for ele in x]
    y = lumen_expert1.diceInfo_2
    
    ax[0].plot(x, y, label = "Lumen Expert 1", marker = "o")
    
    # Plot ORW DSC
    x = list(orw_expert1.diceInfo_1)
    y = orw_expert1.diceInfo_2
    x = [ele.split("_")[-1] for ele in x]
    ax[0].plot(x, y, label = "ORW Expert 1", marker = "o")
    
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
    
    ax[0].plot(x, y, label = "Fat Expert 1", marker = "o")
    
    ax[0].grid()
    ax[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax[0].set_xticklabels(x)
    ax[0].yaxis.set_ticks(np.arange(0, 1, 0.1))
    ax[0].set_xlabel("Slice Number")
    ax[0].set_ylabel("DSC")
    ax[0].set_title(patient)
    
    # PLOT 2
    # Plot lumen DSC
    x = list(lumen_expert2.diceInfo_1)
    x = [ele.split("_")[-1] for ele in x]
    y = lumen_expert2.diceInfo_2
    
    ax[1].plot(x, y, label = "Lumen Expert 2", marker = "o")
    
    # Plot ORW DSC
    x = list(orw_expert2.diceInfo_1)
    x = [ele.split("_")[-1] for ele in x]
    y = orw_expert2.diceInfo_2
    ax[1].plot(x, y, label = "ORW Expert 2", marker = "o")
    
    # Plot fat DSC
    # Some slice of lumen/ORW have no fat.
    # This will cause a mismatch of x andy dimensions on plot
    # Alter fat DSC dataframe to have same number of rows as ORW dataframe
    if(len(x) != fat_expert2.diceInfo_1.shape[0]):
        difference = orw_expert2.loc[~orw_expert2['diceInfo_1'].isin(fat_expert2['diceInfo_1'])]
        difference.loc[difference['diceInfo_2'] > 0.0, "diceInfo_2"] = 0
        
        fat_expert2 = fat_expert2.append(difference, ignore_index = True)
        fat_expert2 = fat_expert2.sort_values(by = ["diceInfo_1"])

    y = fat_expert2.diceInfo_2
    
    ax[1].plot(x, y, label = "Fat Expert 2", marker = "o")

    ax[1].grid()
    ax[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax[1].set_xticklabels(x)
    ax[1].set_xlabel("Slice Number")
    ax[1].set_ylabel("DSC")
    
    fig.savefig("2D Figures/Comparisons/" + patient + ".png", dpi = 300, bbox_inches = "tight")

    