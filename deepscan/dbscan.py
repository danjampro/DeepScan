#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:02:27 2017

@author: danjampro

"""
import time
import numpy as np
from scipy.ndimage.measurements import label, find_objects
from . import minpts, source, geometry, convolution, masking
from .cython import cy_dbscan

#==============================================================================

def ErodeSegmap(segmap, kernel, fft_tol=1E-5):
    '''
    Perform an erosion by kernel on the segmap.
    
    Parameters
    ----------
    segmap : 2D int array
        The segmentation image.
        
    kernel : 2D array
        The kernel to use for the dilation.
        
    fft_tol : float
        The tolerance on the FFT accuracy. Used to surpress artefacts from
        the FFT.
    
    Returns
    -------
    2D int array
        The eroded segmentation image.
    '''        
    #Perform convolution    
    conv = convolution.convolve((segmap==0).astype('float32'), dtype='float32',
                                 kernel=kernel)    
    #Apply threshold                 
    conv = ( conv <= (1+fft_tol) ).astype('int', copy=False)
        
    return conv

#==============================================================================
#Can this class be replaced by a dict?
    
class Clustered():
    '''
    Class to house DBSCAN detections.
    '''
    def __init__(self, corepoints, segmap, segmap_dilate, sources,
                 t_dbscan=None, kernel=None):
        '''
        Parameters
        ----------
        
        '''
        self.corepoints = corepoints
        self.segmap = segmap
        self.sources = sources
        self.segmap_dilate = segmap_dilate
        self.t_dbscan=t_dbscan
        self.kernel = kernel
        
#==============================================================================
    
def _DBSCAN_fft(data, thresh, eps, mpts, erode=True, verbose=True,
                fft_tol=1E-5, get_sources=False, label_segments=True):
    '''
    Perform DBSCAN clustering using a series of thresholding and convolution
    using FFT.
    
    Parameters
    ----------
    data : 2D float array
        Input data 
    
    eps: float
        Clustering radius in pixels.
    
    thresh : float
        Detection threshold [SNR].
        
    mpts : int
        Minimum number of points within eps for clustering.
        
    erode : bool
        If True, generates the segmap from the segmap_dilate.
        
    verbose : bool
        If True, prints information and timings.
        
    fft_tol : float
        The tolerance on the FFT accuracy. Used to surpress artefacts from
        the FFT.
    
    Returns
    -------
    Clustered
        A Clustered object.   
    '''
    t0 = time.time()

    #Create a convolution kernel
    kernel = geometry.unit_tophat(eps)
           
    #Apply the detection threshold                                     
    threshed = data > thresh   
    t_thresh = time.time() - t0
    if verbose: 
        print('-threshold applied in %i seconds' % t_thresh )
        
    #Obtain the core points
    t_dbscan_start = time.time()
    corepts=convolution.convolve(threshed.astype('float32'), kernel=kernel)
    corepts[:, :] = (corepts >= mpts-fft_tol) & threshed
    
    #Recast corepoints to integer 
    corepts = corepts.astype('int')  
    
    ty = time.time() - t_dbscan_start
    if verbose: 
        print('-corepoints obtained in %i seconds' % ty )
                    
    #Obtain the dilated segments
    segmap_dilate =convolution.convolve(corepts.astype('float32'),
                                        kernel=kernel)
    segmap_dilate[:,:] = (segmap_dilate >= 1-fft_tol).astype('int')
    t_sec = time.time() - ty - t_dbscan_start
    if verbose: 
        print('-dilated segments obtained in %i seconds.' % t_sec )
                    
    #Do the labeling & get the slices
    if label_segments:
        segmap_dilate, Nsegments = label(segmap_dilate) 
        corepts[:, :] = corepts * segmap_dilate
        slices = find_objects(segmap_dilate)
        t_labels = time.time()-t_sec-ty-t_dbscan_start
        if verbose: 
            print('-segments labeled in %i seconds.' % t_labels )
    else:
        Nsegments = 0
        slices = None
            
    #Retrieve the Source objects
    if get_sources:
        sources = []
        for i, slice_ in enumerate(slices):
            sources.append( source.Source( i+1, slice_) )
    else:
        sources = None
                
    #Erode the clusters to get the core points
    if erode & (Nsegments != 0):
        if verbose: 
            t_erode = time.time()-t_sec-ty-t_dbscan_start-t_labels
            print('-segmap eroded in %i seconds' % t_erode)
        segmap = ErodeSegmap(segmap_dilate, kernel=kernel, fft_tol=fft_tol)
    else:
        segmap = None
                                                  
    t_dbscan_finish = time.time() - t_dbscan_start
                        
    C = Clustered(corepoints=corepts,segmap=segmap,segmap_dilate=segmap_dilate,
                  sources=sources, t_dbscan=t_dbscan_finish, kernel=kernel) 
    return C


def _DBSCAN(data, thresh, eps, mpts, erode=True, verbose=True,
                fft_tol=1E-5, get_sources=False, label_segments=True):
    '''
    Perform DBSCAN clustering using a series of thresholding and convolution.
    
    Parameters
    ----------
    data : 2D float array
        Input data 
    
    eps: float
        Clustering radius in pixels.
    
    thresh : float
        Detection threshold [SNR].
        
    mpts : int
        Minimum number of points within eps for clustering.
        
    erode : bool
        If True, generates the segmap from the segmap_dilate.
        
    verbose : bool
        If True, prints information and timings.
        
    fft_tol : float
        The tolerance on the FFT accuracy. Used to surpress artefacts from
        the FFT.
            
    Returns
    -------
    Clustered
        A Clustered object.   
    '''
    #Create the convolution kernel
    kernel = geometry.unit_tophat(eps).astype(np.uint8)
           
    #Obtain dilated binary map
    segmap_dilate = cy_dbscan.get_binarymap((data>thresh).astype(np.uint8),
                                             structure=kernel, minpts=mpts)
                   
    #Do the labeling & get the slices
    if label_segments:
        segmap_dilate, Nsegments = label(segmap_dilate) 
        slices = find_objects(segmap_dilate)
    else:
        Nsegments = 0
        slices = None
            
    #Retrieve the Source objects
    if get_sources:
        sources = []
        for i, slice_ in enumerate(slices):
            sources.append( source.Source( i+1, slice_) )
    else:
        sources = None
                
    #Erode the clusters to get the core points
    if erode & (Nsegments != 0):
        segmap = ErodeSegmap(segmap_dilate, kernel=kernel, fft_tol=fft_tol)
    else:
        segmap = None
                                                                          
    C = Clustered(corepoints=None, segmap=segmap, segmap_dilate=segmap_dilate,
                  sources=sources, t_dbscan=None, kernel=kernel) 
    return C

#==============================================================================

def DBSCAN(data, rms, eps=5, thresh=0.5, verbose=True, mask=None, sky=0,
           mpts=None, kappa=5, mask_type='rms', use_fft=True, *args, **kwargs):
    '''
    Run DBSCAN.
    
    Parameters
    ---------
    data : 2D float array
        Input data 
    
    eps: float
        Clustering radius in pixels.
    
    thresh : float
        Detection threshold [SNR].
        
    kappa : float
        Statistical significance parameter. Only used if mpts is None.
    
    mpts : float
        DBSCAN min points parameter. Leave as None to calculate using kappa.
    
    mask_type : string 
        'rms' or 'zeros': How should the mask be applied?
        
    use_fft : bool
        Use FFT convolution instead of direct?
    
    Returns
    -------
    Clustered
        A Clustered object.
    '''
    t0 = time.time()
        
    #Apply the mask
    if mask is not None:
        if mask_type == 'rms':
            data = masking.apply_mask(data, mask=mask, rms=rms, sky=sky)
        elif mask_type == 'zeros':
            data = masking.apply_mask(data, mask=mask, fillval=0)
        else:
            print('WARNING: mask_type not recongnised - ignoring mask.')
    
    #Unit conversions
    data = (data-sky)/rms  #[SNR]
    
    #Calculate minpoints
    if mpts is None:
        mpts = minpts.estimate_minpts(eps=eps, kappa=kappa, tmin=thresh, rms=1)
            
    #DBSCAN clustering
    if verbose: 
        print('dbscan: performing clustering...')
        
    if use_fft:
        C = _DBSCAN_fft(data, thresh=thresh, eps=eps, mpts=mpts,
                        verbose=verbose, *args, **kwargs)
    else:
        C = _DBSCAN(data, thresh=thresh, eps=eps, mpts=mpts,
                    verbose=verbose, *args, **kwargs)
    
    if verbose: 
        print('dbscan: finished after %i seconds.' % (time.time()-t0))
    
    return C
    
#==============================================================================
#==============================================================================

    
        


