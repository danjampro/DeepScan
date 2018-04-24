#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 11:23:18 2017

@author: Dan
"""

import numpy as np
from astropy.io import fits
import os


def read_fits(fname, read_header=False, extension=0, dtype=None):
    
    '''Read fits data'''
    
    #Use astropy to read FITS
    hdulist = fits.open(fname)
    data = hdulist[int(extension)].data
    hdulist.close()
    
    if dtype is not None:
        data = data.astype(dtype)
    
    if read_header:
        header = hdulist[int(extension)].header
        return data, header
    else:
        return data


def save_to_fits(data, fname, header=None, overwrite=False, dtype=None):
    
    '''Save to fits'''
    
    #Check for ofile
    if overwrite == False:
        if os.path.isfile(fname):
            raise(IOError('File %s exists. Set overwrite=True to overwrite.' % fname))
            
    if dtype is None:
        dtype = data.dtype
    
    if dtype == 'bool':
        print('WARNING: Converting bool to int16.')
        dtype = 'int16'
        
    hdulist = fits.PrimaryHDU(data.astype(dtype), header=header)
    hdulist.writeto(fname, overwrite=overwrite)
    
    
def save_to_MEF(datas, headers, fname, overwrite=False):
    
    #Check if file exists
    if overwrite == False:
        if os.path.isfile(fname):
            raise(IOError('File %s exists. Set overwrite=True to overwrite.' % fname))
          
    
    new_hdul = fits.HDUList()
    for data, header in zip(datas, headers):
        new_hdul.append(fits.ImageHDU(data, header=header))
    
    new_hdul.writeto(fname, overwrite=overwrite)