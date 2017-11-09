#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 11:23:18 2017

@author: Dan
"""

import numpy as np
from astropy.io import fits
import os


def read_fits(fname, read_header=False, extension=0):
    
    '''Read fits data'''
    
    #Use astropy to read FITS
    hdulist = fits.open(fname)
    data = hdulist[int(extension)].data
    hdulist.close()
    
    if read_header:
        header = hdulist[int(extension)].header
        return data, header
    else:
        return data


def save_to_fits(data, fname, header=None, overwrite=False, dtype=np.float32):
    
    '''Save to fits'''
    
    #Check for ofile
    if overwrite == False:
        if os.path.isfile(fname):
            raise(IOError('File %s exists. Set overwrite=True to overwrite' % fname))
    
    hdulist = fits.PrimaryHDU(data.astype(dtype), header=header)
    hdulist.writeto(fname, overwrite=overwrite)