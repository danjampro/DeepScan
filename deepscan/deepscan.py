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

def DeepScan(data, verbose=True, makeplots=False, dilate=False,
             kwargs_skymap={}, kwargs_dbscan={}, kwargs_deblend={},
             kwargs_makecat={}):
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
        
    dilate : bool
        Use dilated segmap instead of original?
        
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
    
    #Subtract the sky (this makes a new copy of data)
    data = data-sky
                             
    #Run DBSCAN to identify initial clusters
    C = dbscan.DBSCAN(data=data, rms=rms, verbose=verbose, **kwargs_dbscan)

    #Deblend the segmap produced by DBSCAN 
    segmap_ = C.segmap_dilate if dilate else C.segmap
    segmap, segments = deblend.deblend(data=data, bmap=segmap_, rms=rms,
                                       verbose=verbose, **kwargs_deblend)

    df = makecat.MakeCat(data=data, segmap=segmap, segments=segments,
                         verbose=verbose, **kwargs_makecat)
    
    return {'df':df, 'segmap':segmap, 'sky':sky, 'rms':rms}
    
#==============================================================================
#==============================================================================



