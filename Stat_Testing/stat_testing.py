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
    
Removing outliers:
    https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
    https://matplotlib.org/stable/_images/boxplot_explanation.png
    
    
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ranksums
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import scipy.stats as st
import os

def parse_path(expertpath, segpath):
    """Parse filepath to excel document containing metrics.
    
    Parse a filepath to an excel document containing metrics to get the following information:
        - U-Net type
        - Expert Number (Expert 1 or Expert 2)
        - Metric (Dice, Hausdorff, FD)
    

    Parameters
    ----------
    expertpath : str
        Path to excel document containing metrics for expert 1 vs. expert 2.
    segpath : str
        Path to excel document containing metrics for U-Net vs. Expert.

    Raises
    ------
    ValueError
        Raise error when expertpath and segpath do not have the same U-Net and metric in filename.

    Returns
    -------
    unet : str
        U-Net name.
    expert : str
        Expert 1 or Expert 2.
    metric : str
        Metric name.
    """
    
    def do_parse(path):
        """Sub-function for actually parsing filename.
        
        Parameters
        ----------
        path : str
            Filename to parse.

        Returns
        -------
        unet : str
            U-Net name.
        expert : str
            Expert 1 or Expert 2.
        metric : str
            Metric name.
        """
        thepath = Path(path)
        unet = thepath.parts[7]
        
        if(unet == 'Fat U-Net'):
            expert = thepath.parts[10]
            expert = expert.split("_")[-1]
        else:
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
    """Read excel document with multiple sheets.
    
    Each excel document for expert 1 vs. expert 2 and U-Net contains separate
    sheets for invidual patients. All sheets, except for empty sheets and
    sheets containing median values, are read into 1 DataFrame.
    
    Parameters
    ----------
    excel_path : str
        Filepath to excel document.
    metric : str
        Metric contained in excel document.

    Raises
    ------
    ValueError
        Raises error when the metric is not recognized. If the metric is not
        recognized, sheets containing median values cannot be removed.
    ValueError
        Raises error when the metric is not recognized. If the metric is not
        recognized, the DataFrame column containing metric values cannot be
        converted into a numpy array.

    Returns
    -------
    df : pandas.dataframe
        DataFrame containing 2 columns:
            - 1st column: name of slice
            - 2nd column: metric for the slice
    df_np : numpy.array
        numpy array containing all metric values.

    """
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

def gen_boxplot(expert_data, pred_data, unet, expert, metric, box_path, out_dpi=300, appendString = None):
    """Generate a boxplot of expert vs. U-Net segmentation.
    
    The boxplot contains 2 boxes:
        - Expert 1 vs. Expert 2
        - Expert (1 or 2) vs. U-Net
    
    Parameters
    ----------
    expert_data : numpy.array
        numpy array of metric values corresponding to Expert 1 vs. Expert 2.
    pred_data : numpy.array
        numpy array of metric values corresponding to Expert vs. U-Net.
    unet : str
        U-Net name.
    expert : str
        U-Net segmentation was evaluated on this expert annotation.
    metric : str
        Metric name.
    out_path: str
        Directory to save boxplot
    out_dpi : int, optional
        dpi for output boxplot file. The default is 300.
    appendString : str, optional
        string to append of boxplot title and filename. The default is None.

    Raises
    ------
    ValueError
        Raise error when metric is not recognized. If metric is not recognized,
        ylims cannot be set.

    Returns
    -------
    None.

    """    
    # Set customization options
    boxprops = dict(linewidth=2, color='blue')
    medianprops = dict(linewidth=2.5, color='green') 
    whiskerprops=dict(linewidth=2, color='blue')
    
    # Generate plot title
    if(appendString is not None):
        title = unet + " vs. " + expert + "\n" + metric + appendString
        filename = box_path + unet + '_' + metric + '_' + expert + appendString + '.png'
    else:
        title = unet + " vs. " + expert + "\n" + metric
        filename = box_path + unet + '_' + metric + '_' + expert + '.png'
    
    # Create figure
    fig, ax = plt.subplots()
    ax.boxplot([expert_np, pred_np], labels=["Expert 1 vs. Expert 2",unet + " vs. " + expert], boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops)
    ax.set_title(title)
    
    if(metric == "Dice"):
        ax.set_ylim((0,1))
    elif(metric == "Hausdorff"):
        ax.set_ylim((0, 6))
    elif(metric == "FD"):
        ax.set_ylim((0, 6))
    else:
        raise ValueError("Metric not recognized!")
        
    ax.set_ylabel(metric)
    
    # Save figure
    fig.savefig(filename, bbox_inches='tight', dpi=out_dpi)
    plt.close()
    
def calculate_percentile(data, percentile):
    """Calculate the nth percentile.
    
    This function will remove outliers outside of nth percentile.

    Parameters
    ----------
    data : numpy.array
        Array of data over which to compute the nth percentile.
    percentile : int
        Percentile to calculate.

    Returns
    -------
    new_data : numpy.array
        Data without outliers outside nth percentile.
    """
    maximum = np.percentile(data, percentile, 0, keepdims=True)
    new_data = data [data <= maximum]
    return new_data

def calculate_IQR(data):
    """Calculate IQR for a dataset to remove outliers.
    
    This function will remove outliers below and above the interquartile
    range (IQR).

    Parameters
    ----------
    data : numpy.array
        Array of data over which to remove outliers outside of IQR.

    Returns
    -------
    new_data : numpy.array
        Data without outliers outside IQR.
    IQR : float
        Calculated IQR of data.

    """
    Q3, Q1 = np.percentile(data, [75 ,25])
    IQR = Q3 - Q1
    new_data = data[ (data > (Q1 - 1.5 * IQR)) & (data < (Q3 + 1.5 * IQR))]
    return new_data, IQR
    

def write_table(data, row_length):
    """Write pandas dataframe as HTML table.

    Parameters
    ----------
    data : pandas.dataframe
        Dataframe to convert to HTML table.
    row_length : int
        Length of each HTML table row.

    Returns
    -------
    out : str
        HTML code for table.

    """
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

def bold_significant(val, cutoff):
    """Bold statistically significant values in pandas dataframe.
    
    Specify the style of a pandas dataframe to bold values that are less than
    the cutoff value for statistical significance.
    
    Parameters
    ----------
    val : pandas.dataframe cell
        Value of dataframe cell to determine if it should be bold based on
        its value.
    cutoff : float
        Cutoff value for statistical significance.

    Returns
    -------
    pandas.style
        Updated style of pandas dataframe.
    """
    bold = 'bold' if val < cutoff else ''
    return 'font-weight: %s' % bold

# Uncomment these for no outlier removal, removing outliers outside of 90th percentile, or removing outliers outside of IQR
append = None
# append = "_90th_Percentile"
# append = "_IQR"

# Specify output directories
outpath = '/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/Stat_Testing/2022_04_27_Results/'
boxplot_out = outpath + 'Boxplots/'

if not os.path.isdir(outpath):
    os.mkdir(outpath)
    
if not os.path.isdir(boxplot_out):
    os.mkdir(boxplot_out)

# Open text and HTML files that will store statistical testing results
if(append is not None):
    stat_file = open(outpath + "Statistical_Testing_Results" + append + ".txt", "w")
    HTML_File=open(outpath + 'Statistical_Testing_Results' + append + '.html','w')
else:
    stat_file = open(outpath + "Statistical_Testing_Results.txt", "w")
    HTML_File=open(outpath + 'Statistical_Testing_Results.html','w')

# Create dataframe that will hold statistical testing results
output_table = pd.DataFrame(columns=["U-Net", "Compared To", "Metric", "p-Value", "Statistic", "Significance", "Confidence"])

# Open document with all fielapths
filepaths = pd.read_excel("/Volumes/GoogleDrive/My Drive/tom/Rectal Segmentation/Data-MultipleExperts/Figures/Stat_Testing/2022_04_27_Stat_Filepaths.xlsx")

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

    if(append is None):
        # Generate boxplot
        gen_boxplot(expert_np, pred_np, unet, expert, metric, boxplot_out)
        
    if(append == "_90th_Percentile"):
        # Calculate 90th percentile and generate boxplots
        expert_np = calculate_percentile(expert_np, 90)
        pred_np = calculate_percentile(pred_np, 90)
        gen_boxplot(expert_np, pred_np, unet, expert, metric, boxplot_out, appendString=append)
        
    if(append == "_IQR"):
        # Calculate IQR and generate boxplots
        expert_np, expert_IQR = calculate_IQR(expert_np)
        pred_np, expert_IQR = calculate_IQR(pred_np)
        gen_boxplot(expert_np, pred_np, unet, expert, metric, boxplot_out, appendString=append)
    
    # Compute p-value via Wilcoxon Ranksum
    statistic, pval = ranksums(expert_np, pred_np)
    
    # Compute p-value via t-test
    #statistic, pval = ttest_ind(expert_np, pred_np)
    
    # Calculate confidence interval
    interval = st.t.interval(alpha=0.95, df=len(pred_np)-1, loc=np.mean(pred_np), scale=st.sem(pred_np))
    
    # Write results to text file
    out_text = "U-Net: %s \nExpert: %s \nMetric: %s \np-Value: %s \nStatistic: %s \nInterval: %s \n\n" % (unet, expert, metric, pval, statistic, interval)
    all_text_outputs.append(out_text)
    stat_file.write(out_text)
    
    # Determine significance
    # NOTE: the cut-off for significance is 0.05 / 6 comparisons = 0.008
    cutoff = 0.008
    if(pval < cutoff):
        significance = True
    else:
        significance = False
    
    # Create temporary dataframe and merge with output dataframe
    temp_df = {'U-Net': unet, 'Compared To': expert, 'Metric': metric, 'p-Value': pval, 'Statistic':statistic, "Significance":significance, "Confidence": interval}
    output_table = output_table.append(temp_df, ignore_index = True)

# Apple some styling to output dataframe
s = output_table.style.set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'yellow')]}])
s.format('{0:,.20f}', subset=['p-Value', 'Statistic'])
s.applymap(lambda x: bold_significant(x, cutoff), subset=['p-Value'])

# Write the formatted HTML dataframe to HTML file    
HTML_File.write(s.render())

# Close files
stat_file.close()
HTML_File.close()
    

    






