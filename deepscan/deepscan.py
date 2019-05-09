#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:42:14 2019

@author: danjampro

This file contains the DeepScan macro function:
    
    skymap > dbscan > deblend > makecat
"""
from . import skymap, dbscan, deblend, makecat

#==============================================================================

def DeepScan(data, verbose=True, makeplots=False, kwargs_skymap={},
             kwargs_dbscan={}, kwargs_deblend={}, kwargs_makecat={}):
    '''
    Run the DeepScan pipeline with default arguments, unless otherwise 
    specified.
    
    Parameters
    ----------
    data : 2D float array
        The data.
        
    verbose : bool
        Print information?
        
    makeplots : bool
        Make check plot(s)?
        
    kwargs_skymap : dict
        Keyword arguments to be passed to skymap.skymap.
        
    kwargs_dbscan : dict
        Keyword arguments to be passed to dbscan.DBSCAN.
        
    kwargs_deblend : dict
        Keyword arguments to be passed to deblend.deblend.
        
    kwargs_makecat : dict
        Keyword arguments to be passed to makecat.MakeCat.
    
    Returns
    -------
    dict
        A dictionary containing the catalogue, segmap, sky, rms and sources.
    '''
    #Measure the sky and its RMS
    sky, rms = skymap.skymap(data=data, verbose=verbose, **kwargs_skymap)
                             
    #Run DBSCAN to identify initial clusters
    C = dbscan.DBSCAN(data=data, sky=sky, rms=rms, verbose=verbose,
                      **kwargs_dbscan)

    #Deblend the segmap produced by DBSCAN 
    segmap, sources = deblend.deblend(data=data, bmap=C.segmap, rms=rms,
                                      verbose=verbose, **kwargs_deblend)

    cat = makecat.MakeCat(data=data, segmap=segmap, sources=sources,
                          verbose=verbose, **kwargs_makecat)
    
    return {'cat':cat, 'segmap':segmap, 'sky':sky, 'rms':rms,
            'sources':sources}
    
#==============================================================================
#==============================================================================



