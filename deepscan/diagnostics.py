#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:04:07 2017

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from . import dbscan

def ellipse_plot(fitsdata, cfile, ax=None, **kwargs):
    
    '''Plot arcsinh of data and overlay ellipses'''
    
    #Read data from fits
    h = fits.open(fitsdata)
    data = h[0].data
    h.close()
    
    #Read ellipses from cluster file
    ellipses = dbscan.read_ellipses(cfile)
    
    #Show the result
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow( np.arcsinh(data), cmap='binary')
    [e.draw(ax, **kwargs) for e in ellipses]
    shape = data.shape
    plt.xlim(0, shape[1])
    plt.ylim(0, shape[0])