#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:02:08 2017

@author: danjampro
"""

import os, multiprocessing, tempfile
from functools import partial
import numpy as np
from scipy.signal import fftconvolve
from . import NTHREADS, BUFFSIZE


def perform_convolution(xmin, xmax, ymin, ymax, R, kernel, dshape):
    '''
    Convolve a portion of the data using fftconvolve.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
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
   
    
def conv_init(data, conv):
    '''
    Initializer for convolve_large process pool.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    global data_, conv_
    data_ = data; conv_ = conv
    
def convolve_large(data, kernel, meshsize=None, Nthreads=None, dtype=None):
    '''
    Convolve a large image, returning a memmap.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    if Nthreads is None:
        Nthreads = NTHREADS
        
    if meshsize is None:
        meshsize = int( BUFFSIZE / data.dtype.itemsize) 
        
    if dtype is None:
        dtype = data.dtype
        
    pool = None      
    with tempfile.TemporaryDirectory() as temppath:                                         
        try:
            
            #Create a memory map for the convolved image
            convfilename = os.path.join(temppath, 'conv.memmap')
            conv_memmap = np.memmap(convfilename, dtype=dtype, mode='w+',
                                        shape=data.shape)
            
            #Create the chunk boundary arrays
            xmins = np.arange(0, data.shape[1], meshsize)
            xmaxs = np.arange(meshsize, data.shape[1]+meshsize, meshsize) 
            ymins = np.arange(0, data.shape[0], meshsize) 
            ymaxs = np.arange(meshsize, data.shape[0]+meshsize, meshsize)
            bounds_list = [[xmins[i],xmaxs[i],ymins[j],ymaxs[j]]
                            for i in range(xmins.size) for j in range(ymins.size)]
            
            #R is the size of the expansion required for continuity
            R = int(np.max((kernel.shape[0], kernel.shape[1])))
            
            if Nthreads > 1:
                
                #Create process pool
                pool = multiprocessing.Pool(processes=Nthreads,initializer=conv_init,
                                            initargs=(data, conv_memmap))
                        
                #Create a function to perform the convoluton
                pfunc = partial(perform_convolution, kernel=kernel,
                                 dshape=data.shape, R=R)                      
                #Do the conv
                pool.starmap(pfunc, bounds_list)
                
            else:   #Single thread mode - does the same as perform_convolution
            
                for b in bounds_list:
                    #Expand box
                    xmax2 = b[1] + R
                    xmin2 = b[0] - R
                    ymin2 = b[2] - R
                    ymax2 = b[3] + R
                    
                    #Look for boundary overlap
                    xoverlap1 = np.max((0, -xmin2))          
                    xoverlap2 = np.max((0, xmax2-data.shape[1]))  
                    yoverlap1 = np.max((0, -ymin2))           
                    yoverlap2 = np.max((0, ymax2-data.shape[0]))  
                    
                    #Crop
                    xmax2 = int(np.min((xmax2, data.shape[1])))
                    ymax2 = int(np.min((ymax2, data.shape[0])))
                    xmin2 = int(np.max((xmin2, 0)))
                    ymin2 = int(np.max((ymin2, 0)))
                      
                    cnv = fftconvolve(np.array(data[ymin2:ymax2,xmin2:xmax2]),
                                                        kernel, mode='same')
                        
                    conv_memmap[b[2]:b[3], b[0]:b[1]] = cnv[
                                            R-yoverlap1:cnv.shape[0]-R+yoverlap2,
                                            R-xoverlap1:cnv.shape[1]-R+xoverlap2]
        finally:
            if pool is not None:
                pool.close()
                pool.join()       
    return conv_memmap
        
     

def convolve(data, kernel, lowmem=False, meshsize=None, Nthreads=None,
             dtype=None):
    '''
    Perform FFT convolution.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    if dtype is None:
        dtype = data.dtype
        
    if lowmem:
        return convolve_large(data, kernel, meshsize=meshsize,
                              Nthreads=Nthreads, dtype=dtype) 
    else:
        return fftconvolve(np.array(data), kernel, mode='same').astype(dtype)
                                        
