#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:30:31 2021

@author: Tom

Styling with Pandas:
    - https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Table-Styles
    - https://pbpython.com/styling-pandas.html
    
Computing Confidence Interval:
    - https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
"""

import os
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ranksums
from scipy.stats import ttest_ind
import scipy.stats as st

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

def write_table(data, row_length):
    out = '<table>'
    counter = 0
    for element in data:
        if counter % row_length == 0:
            out += '<tr>'
        out += ('<td>%s</td>' % element)
        counter += 1
        if counter % row_length == 0:
            out += '</tr>'
    if counter % row_length != 0:
        for i in range(0, row_length - counter % row_length):
            out += '<td>&nbsp;</td>'
        out += '</tr>'
    out += '</table>'
    return out

def bold_significant(val):


    bold = 'bold' if val < 0.05 else ''
    return 'font-weight: %s' % bold

# Open text and HTML files that will store statistical testing results
stat_file = open("Statistical_Testing_Results.txt", "w")
HTML_File=open('Statistical_Testing_Results.html','w')

# Create dataframe that will hold statistical testing results
output_table = pd.DataFrame(columns=["U-Net", "Compared To", "Metric", "p-Value", "Statistic", "Significance", "Confidence"])

# Open document with all fielapths
filepaths = pd.read_excel("/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/Stat_Filepaths.xlsx")

# Create lists of expert and prediction filepaths
expert_paths = filepaths["expert"].to_list()
expert_paths = [s.replace("\ufeff","") for s in expert_paths]
pred_paths = filepaths["unet"].to_list()
pred_paths = [s.replace("\ufeff","") for s in pred_paths]

# Store output text of statistical testing
all_text_outputs = []

for index, path in enumerate(expert_paths):
    expert_excel_path = path
    pred_excel_path = pred_paths[index]

    # Parse path for unet, expert number, and metric
    unet, expert, metric = parse_path(expert_excel_path, pred_excel_path)
    
    # print("U-Net: %s \n Expert: %s \n Metric: %s \n" % (unet, expert, metric))

    # Load in the metrics
    expert_df, expert_np = load_excel(expert_excel_path, metric)
    pred_df, pred_np = load_excel(pred_excel_path, metric)

    # Compute p-value via Wilcoxon Ranksum
    statistic, pval = ranksums(expert_np, pred_np)
    #statistic, pval = ttest_ind(expert_np, pred_np)
    interval = st.t.interval(alpha=0.95, df=len(pred_np)-1, loc=np.mean(pred_np), scale=st.sem(pred_np))
    
    # Write results to text file
    out_text = "U-Net: %s \nExpert: %s \nMetric: %s \np-Value: %s \nStatistic: %s \nInterval: %s \n\n" % (unet, expert, metric, pval, statistic, interval)
    all_text_outputs.append(out_text)
    stat_file.write(out_text)
    
    # Determine significance
    if(pval < 0.05):
        significance = True
    else:
        significance = False
    
    # Create temporary dataframe and merge with output dataframe
    temp_df = {'U-Net': unet, 'Compared To': expert, 'Metric': metric, 'p-Value': pval, 'Statistic':statistic, "Significance":significance, "Confidence": interval}
    output_table = output_table.append(temp_df, ignore_index = True)

# Apple some styling to output dataframe
s = output_table.style.set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'yellow')]}])
s.format('{0:,.20f}', subset=['p-Value', 'Statistic'])
s.applymap(bold_significant,subset=['p-Value'])

# Write the formatted HTML dataframe to HTML file    
HTML_File.write(s.render())

# Close files
stat_file.close()
HTML_File.close()
    

    






