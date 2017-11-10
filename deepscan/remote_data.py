#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:09:24 2017

@author: danjampro
"""

import tempfile, os, shutil, urllib, gzip
from . import utils

def get(url):
    
    '''Download XXX.fits.gz file and read in memory'''
    
    tdir = tempfile.mkdtemp() 
    
    try:
        
        fname_gz = os.path.join(tdir, 'ngvs_small.fits.gz')
        fname = os.path.join(tdir, 'ngvs_small.fits')
        urllib.request.urlretrieve(url, fname_gz)
        
        with gzip.open(fname_gz, 'rb') as f_in, open(fname, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
        data = utils.read_fits(fname)
        
    finally:
        shutil.rmtree(tdir)
        
    return data