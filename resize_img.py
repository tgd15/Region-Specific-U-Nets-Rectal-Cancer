#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:22:08 2019

@author: Tom
"""

import matplotlib.pyplot as plt
from PIL import Image
import cv2

def resize_img(img_slice, new_size, nn=False, plot=False, filepath=None, filename=None):
    """Resize an image with optional plotting.
    
    Resize an image to a desired size. The original image and resized image can
    be plotted side-by-side if specified.
    

    Parameters
    ----------
    img_slice : np.ndarray
        Image to resize.
    new_size : tuple
        New size of image.
    nn : bool, optional
        Specify nearest neighbor interpolation when resizing images. If true,
        nearest neighbor interpolation will be used. Specify true if resizing mask.
        The default is False.
    plot : bool, optional
        Plot original image and resized image side-by-side. The default is False.
    filepath : str, optional
        Root path into directory where plot will be saved. The default is None.
    filename : str, optional
        Title of plot. The default is None.

    Returns
    -------
    resized_img : TYPE
        DESCRIPTION.

    """
    if(nn is False):
        resized_img = cv2.resize(img_slice, dsize=new_size, interpolation=Image.LINEAR) # Reize cropped image with linear interpolation
    else:
        resized_img = cv2.resize(img_slice, dsize=new_size, interpolation=Image.NEAREST) # Reize cropped image with linear interpolation
    
    if(plot == True):
        fig, ax = plt.subplots(1, 2, figsize=(50, 10.0))
        
        # Original Image
        ax[0].set_title('Original Image')
        ax[0].imshow(img_slice, cmap='gray')
        
        # Diplay image that was cropped to bounding box and resized
        ax[1].set_title('Resized ' + filename)
        ax[1].imshow(resized_img, cmap='gray')
        
        # Save the figure
        fig.savefig(filepath+filename+'-crop.png', bbox_inches='tight')
    return resized_img