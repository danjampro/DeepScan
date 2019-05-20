#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:09:24 2017

@author: danjampro

Get some test data from the github repo.
"""
import tempfile, os, shutil, urllib, gzip
from . import utils, __file__

#==============================================================================

def get(i=1):
    '''
    Download XXX.fits.gz to a tempfile and open as np.array. 
    
    Parameters
    ----------
    i : int
        Identifier for the data.
    
    Returns
    -------
    2D float array
        The data.
    '''
    #Check for local data
    fname = 'testimage%i.fits' % (i)
    path = os.path.dirname((os.path.dirname(__file__)))
    fname_full = os.path.join(path, 'data', fname)
    if os.path.isfile(fname_full):
        return utils.read_fits(fname_full).astype('float32')
        
    #Download the data to this directory
    tdir = tempfile.mkdtemp() 
    
    #This is the download ur;
    url = 'https://raw.github.com/danjampro/DeepScan/master/data/'
    url+= '%s.gz' % (fname)
    
    try:
        #Download the data
        fname_gz = os.path.join(tdir, 'testimage%i.fits.gz' % i)
        fname = os.path.join(tdir, 'testimage%i.fits' % i)
        urllib.request.urlretrieve(url, fname_gz)
        
        #Unzip the data
        with gzip.open(fname_gz, 'rb') as f_in, open(fname, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
        #Read into Python
        data = utils.read_fits(fname).astype('float32')
        
    finally:        
        #Delete the data on disk
        shutil.rmtree(tdir)
        
    return data

#==============================================================================
#==============================================================================