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
from scipy.stats import ttest_ind

def parse_path(expertpath, segpath):
    
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
    elif(ex_unet == "Rectal Wall U-Net" and seg_unet == "Segmentation Fusion" and ex_metric == seg_metric):
        unet = seg_unet
        expert = seg_expert
        metric = ex_metric
    else:
        raise ValueError("Filepaths do not correspond to each other!")
    
    return unet, expert, metric

def load_excel(excel_path, metric):
    df = pd.concat(pd.read_excel(excel_path, sheet_name=None), ignore_index=True)
    
    if(metric == "Dice"):
        df = df.drop(["medianDiceCell1", "medianDiceCell2"],axis="columns")
    elif(metric == "Hausdorff" or metric=="HD"):
        df = df.drop(["medianHausCell1", "medianHausCell2"],axis="columns")
    elif(metric == "Frechet" or metric == "FD"):
        df = df.drop(["medianFDCell1", "medianFDCell2"],axis="columns")
    else:
        raise ValueError("Median metric not recognized!")
        
    df = df.dropna()
    
    if(metric == "Dice"):
        df_np = df["diceInfo_2"].to_numpy()
    elif(metric == "Hausdorff" or metric=="HD"):
        df_np = df["HDInfo_2"].to_numpy()
    elif(metric == "Frechet" or metric == "FD"):
        df_np = df["FDInfo_2"].to_numpy()
    else:
        raise ValueError("Metric not recognized!")
        
    return df, df_np

stat_file = open("Statistical_Testing_Results.txt", "w")

filepaths = pd.read_excel("/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/Stat_Filepaths.xlsx")
expert_paths = filepaths["expert"].to_list()
expert_paths = [s.replace("\ufeff","") for s in expert_paths]
pred_paths = filepaths["unet"].to_list()
pred_paths = [s.replace("\ufeff","") for s in pred_paths]

all_text_outputs = []

for index, path in enumerate(expert_paths):
    expert_excel_path = path
    pred_excel_path = pred_paths[index]

    # Parse path for unet, expert number, and metric
    unet, expert, metric = parse_path(expert_excel_path, pred_excel_path)
    
    print("U-Net: %s \n Expert: %s \n Metric: %s \n" % (unet, expert, metric))

    # Load in the metrics
    expert_df, expert_np = load_excel(expert_excel_path, metric)
    pred_df, pred_np = load_excel(pred_excel_path, metric)

    # Compute p-value via Wilcoxon Ranksum
    #statistic, pval = ranksums(expert_np, pred_np)
    statistic, pval = ttest_ind(expert_np, pred_np)
    
    # Write results to text file
    out_text = "U-Net: %s \nExpert: %s \nMetric: %s \np-Value: %s \nStatistic: %s \n\n" % (unet, expert, metric, pval, statistic)
    all_text_outputs.append(out_text)
    stat_file.write(out_text)

stat_file.close()

def print_table(data, row_length):
    print ('<table>')
    counter = 0
    for element in data:
        if counter % row_length == 0:
            print ('<tr>')
        print ('<td>%s</td>' % element)
        counter += 1
        if counter % row_length == 0:
            print ('</tr>')
    if counter % row_length != 0:
        for i in range(0, row_length - counter % row_length):
            print ('<td>&nbsp;</td>')
        print ('</tr>')
    print ('</table>')
    
HTML_File=open('stats.html','w')
HTML_File.write(str(print_table(all_text_outputs,1)))
HTML_File.close()



