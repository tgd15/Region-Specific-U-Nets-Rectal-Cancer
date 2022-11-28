#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:30:55 2020

@author: Tom
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def bounding_box(mask_slice, img_slice, size, filepath, filename, plot=False):
    """Create a bounding box around the biggest component in a binary 2D image
    
    For this function to work properly, the mask MUST be binary (i.e. only 1 non-zero value is within mask):

    Parameters
    ----------
    mask_slice : array
        Binary 2D array
        
    img_slice : array
        Binary 2D array
        
    size : int
        Size of bounding box
        
    filepath : str
        Root path to save plots
        
    filename : str
        Name of plots
        
    plot : bool, optional
        Defaults to False. If True, plot bounding box immediately around largest component and expanded bounding box around largest component
        Saves plot as `filename` to `filepath` 

    Returns
    -------
    ex_bbox_mask : array
        Binary 2D array of mask cropped to expanded bounding box
        
    ex_bbox_img : array
        Binary 2D array of image cropped to expanded bounding box

        
    """

    # Get indices of unique, non-zero values in mask 
    height, width = np.nonzero(mask_slice)
    height = np.unique(height)
    width = np.unique(width)
    
    
    '''
    Only uncomment for debugging purposes
    -------------------------------------
    '''
    # Generate bounding box
    # Numpy slicing occurs on y-axis (height) first, then on x-axis(width)
    #bbox_mask = mask_slice[min(height):max(height),min(width):max(width)]
    #bbox_img = img_slice[min(height):max(height),min(width):max(width)]
    '''
    -------------------------------------
    '''
    
    # Expand bbox by this number of pixels
    ex_num = size
    
    bottom = min(max(height)+ex_num,mask_slice.shape[0]);
    top = max(min(height)-ex_num,0); 
    left = max(min(width)-ex_num,0); 
    right = min(max(width)+ex_num,mask_slice.shape[1]);
    
    # Crop mask and image to expanded bbox
    ex_bbox_mask = mask_slice[top:bottom, left:right]
    ex_bbox_img = img_slice[top:bottom, left:right]
    
    if(plot==True):
        # Display bounding box around rectum
        fig, ax = plt.subplots(1, 2, figsize=(50, 20))
        ax[0].set_title('Rectal Wall Bounding Box')
        ax[0].imshow(img_slice, cmap='gray')
        ax[0].add_patch(patches.Rectangle((min(width),min(height)),(max(width)-min(width)),(max(height)-min(height)),alpha=0.6, label='Bounding Box around Rectum'))
        ax[0].contour(mask_slice,colors='y', levels=[0.5])
        ax[0].legend()
        
        # Display expanded bounding box around rectum
        ax[1].set_title('Expanded Bounding Box')
        ax[1].imshow(img_slice, cmap='gray')
        ax[1].contour(mask_slice,colors='y', levels=[0.5])
        ax[1].add_patch(patches.Rectangle((left,top),(right-left),(bottom-top),alpha=0.3, color='r', label='Expanded Bounding Box around Rectum'))
        ax[1].legend()
        
        fig.savefig(filepath+filename+'-bbox.png', bbox_inches='tight')
    return ex_bbox_mask, ex_bbox_img

