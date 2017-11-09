#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:02:08 2017

@author: danjampro
"""

import os, multiprocessing, tempfile, shutil
from functools import partial
import numpy as np
from scipy.signal import fftconvolve
from . import NTHREADS, BUFFSIZE


def convolve_large(data, kernel, meshsize=None, Nthreads=None):
    
    if Nthreads is None:
        Nthreads = NTHREADS
        
    if meshsize is None:
        meshsize = int( BUFFSIZE / data.dtype.itemsize) 
        
    temppath = tempfile.mkdtemp()
    pool = None                                               
    try:
        
        #Create a memory map for the convolved image
        convfilename = os.path.join(temppath, 'conv.memmap')
        conv_memmap = np.memmap(convfilename, dtype='float32', mode='w+',
                                    shape=data.shape)
        
        #Create process pool
        pool = multiprocessing.Pool(processes=Nthreads,initializer=conv_init,
                                    initargs=(data, conv_memmap))
    
        #Create the chunk boundary arrays
        xmins = np.arange(0, data.shape[1], meshsize)
        xmaxs = np.arange(meshsize, data.shape[1]+meshsize, meshsize) 
        ymins = np.arange(0, data.shape[0], meshsize) 
        ymaxs = np.arange(meshsize, data.shape[0]+meshsize, meshsize)
        bounds_list = [[xmins[i],xmaxs[i],ymins[j],ymaxs[j]]
                        for i in range(xmins.size) for j in range(ymins.size)]
        
        #Create a function to perform the convoluton
        pfunc = partial(perform_convolution, kernel=kernel,
                         dshape=data.shape, R=int(np.max((kernel.shape[0], kernel.shape[1]))))                      
        #Do the conv
        pool.starmap(pfunc, bounds_list)
        
        #Save the array
        conv = np.array(conv_memmap)
        
    finally:
        del conv_memmap
        shutil.rmtree(temppath)
        if pool is not None:
            pool.close()
            pool.join()
            
    return conv
        
        
        
def conv_init(data, conv):
    global data_, conv_
    data_ = data; conv_ = conv
def perform_convolution(xmin, xmax, ymin, ymax, R, kernel, dshape):
    
    #Expand box
    xmax2 = xmax + R
    xmin2 = xmin - R
    ymin2 = ymin - R
    ymax2 = ymax + R
    
    #Look for boundary overlap
    xoverlap1 = np.max((0, -xmin2))           #Negative x boundary overlap
    xoverlap2 = np.max((0, xmax2-dshape[1]))  #Positive x boundary overlap
    yoverlap1 = np.max((0, -ymin2))           #Negative y boundary overlap
    yoverlap2 = np.max((0, ymax2-dshape[0]))  #Positive y boundary overlap
    
    #Crop
    xmax2 = int(np.min((xmax2, dshape[1])))
    ymax2 = int(np.min((ymax2, dshape[0])))
    xmin2 = int(np.max((xmin2, 0)))
    ymin2 = int(np.max((ymin2, 0)))
      
    cnv = fftconvolve(np.array(data_[ymin2:ymax2,xmin2:xmax2]),
                                        kernel, mode='same')
        
    conv_[ymin:ymax, xmin:xmax] = cnv[R-yoverlap1:cnv.shape[0]-R+yoverlap2,
                                            R-xoverlap1:cnv.shape[1]-R+xoverlap2]