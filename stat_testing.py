#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:30:31 2021

@author: Tom
"""

import os
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ranksums

def parse_path(expertpath, segpath):
    """
    

    Parameters
    ----------
    expertpath : TYPE
        DESCRIPTION.
    segpath : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    unet : TYPE
        DESCRIPTION.
    expert : TYPE
        DESCRIPTION.
    metric : TYPE
        DESCRIPTION.

    """
    
    def do_parse(path):
        thepath = Path(path)
        unet = thepath.parts[7]
        expert = thepath.parts[9]
        expert = expert.split("_")[-1]
        metric = thepath.parts[-1]
        metric = metric.split("_")[0]
        return unet, expert, metric
    
    ex_unet, _, ex_metric = do_parse(expertpath)
    seg_unet, seg_expert, seg_metric = do_parse(segpath)
    
    if(ex_unet == seg_unet and ex_metric == seg_metric):
        unet = ex_unet
        expert = seg_expert
        metric = ex_metric
    else:
        raise ValueError("Filepaths do not correspond to each other!")
    
    return unet, expert, metric

def load_excel(excel_path, metric):
    df = pd.concat(pd.read_excel(excel_path, sheet_name=None), ignore_index=True)
    
    if(metric == "Dice"):
        df = df.drop(["medianDiceCell1", "medianDiceCell2"],axis="columns")
    elif(metric == "Hausdorff"):
        df = df.drop(["medianHausCell1", "medianHausCell2"],axis="columns")
    elif(metric == "Frechet"):
        df = df.drop(["medianFDCell1", "medianFDCell2"],axis="columns")
    else:
        raise ValueError("Median metric not recognized!")
        
    df = df.dropna()
    
    if(metric == "Dice"):
        df_np = df["diceInfo_2"].to_numpy()
    elif(metric == "Hausdorff"):
        df_np = df["HDInfo_2"].to_numpy()
    elif(metric == "Frechet"):
        df_np = df["FDInfo_2"].to_numpy()
    else:
        raise ValueError("Metric not recognized!")
        
    return df, df_np


expert_excel_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Outer Rectal Wall U-Net/Results/ExperttoExpert/Dice_Per_Slice.xlsx"

pred_excel_path = "/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Outer Rectal Wall U-Net/Results/CCA_Expert1/Dice_Per_Slice.xlsx"

# Parse path for unet, expert number, and metric
unet, expert, metric = parse_path(expert_excel_path, pred_excel_path)

# Load in the metrics
expert_df, expert_np = load_excel(expert_excel_path, metric)
pred_df, pred_np = load_excel(pred_excel_path, metric)

# Compute p-value via Wilcoxon Ranksum
statistic, pval = ranksums(expert_np, pred_np)

pd.DataFrame(columns=["col1","col2","col3"])

stat_file = "Statistical_Testing_Results.txt"

