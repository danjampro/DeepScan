#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:02:08 2017

@author: danjampro
"""
from scipy.signal import fftconvolve

#==============================================================================    

def convolve(data, kernel, dtype=None):
    '''
    Perform FFT convolution. 
    
    Paramters
    ---------
    data : 2D float array
        The data array.
        
    dtype : type
        The type of the convolved array output.
    
    Returns
    -------
    2D float array
        The convolved array.
    '''
    if dtype is None:
        dtype = data.dtype
    
    #Do the convolution
    return fftconvolve(data, kernel, mode='same').astype(dtype, copy=False)
    
#==============================================================================

                                        
