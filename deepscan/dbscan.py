#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:02:27 2017

@author: danjampro

"""

import io, os, tempfile, time
import numpy as np
from scipy.ndimage.measurements import label, find_objects
from . import minpts, source, geometry, convolution, masking, NTHREADS

#==============================================================================

def erode_segmap(segmap, kernel, meshsize=None, lowmem=False, Nthreads=NTHREADS,
                 fft_tol=1E-5):
    '''
    Perform an erosion by kernel on the segmap.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    if meshsize is None:
        meshsize = int(io.DEFAULT_BUFFER_SIZE/4) #32 bit floating points
        
    if lowmem:
        with tempfile.TemporaryDirectory() as tempdir: 
        
            arr = np.memmap(os.path.join(tempdir, 'temp.dat'),
                            shape=segmap.shape, mode='w+', dtype='float32')                     
            arr[:,:] = (segmap==0).astype('float32')
            conv = convolution.convolve(arr, lowmem=lowmem,
                                              kernel=kernel, Nthreads=Nthreads,
                                              meshsize=meshsize,dtype='float32')
                                               
    else:   #High memory mode
        conv = convolution.convolve((segmap==0).astype('float32'), dtype='float32',
                                 kernel=kernel, lowmem=False)
    #Housekeeping                              
    conv[:,:] = conv <= fft_tol
    conv[:,:] = conv.astype('int') * segmap

    return conv
    
    
#==============================================================================
    
def dbscan_conv(data, thresh, eps, mpts, Nthreads=NTHREADS,
                                         meshsize=None,
                                         thresh_type='SNR',
                                         rms=None,
                                         erode=True,
                                         lowmem=False,
                                         verbose=True,
                                         fft_tol=1E-5):
                                             
    #Do some argument checking
    if Nthreads < 1:
        raise ValueError('Nthreads<1.')

    if Nthreads > 1:
        if meshsize is not None:
            try:
                assert(type(meshsize)==int) 
            except AssertionError:
                raise ValueError('meshsize must be an integer.')
            try:
                assert(meshsize>0)
            except AssertionError:
                raise ValueError('meshsize must be >0.')
    
    try:
        assert( (type(eps)==int) or (type(eps)==float) )
    except:
        raise TypeError('eps parameter should either be an integer or double.')
    
    try:
        assert(type(mpts)==int)
    except:
        raise TypeError('mpts parameter should either be an integer or double.')
    
    if thresh_type == 'absolute':
        try:
            assert(rms is None)
            rms = 1
        except:
            TypeError('rms map should be None if thresh_type=absolute.')
    elif thresh_type == 'SNR':
        try:
            assert(rms is not None)
        except:
            TypeError('rms map should not be None if thresh_type=SNR.')
    else:
        raise ValueError("Allowable thresh_type(s): 'absolute', 'SNR'.")
        
                   
    #Create a convolution kernel
    kernel_conv = geometry.unit_tophat(eps)
    
    #Create a tempory file for the memory map
    temppath = tempfile.mkdtemp()
    pool = None  
    t0 = time.time()    
    with tempfile.TemporaryDirectory() as temppath:                                          
    
        try:
            #Obtain the thresholded image
            if lowmem:
                threshfilename = os.path.join(temppath, 'temp1.memmap')
                threshed = np.memmap(threshfilename, dtype='int', mode='w+',
                                                         shape=data.shape)
                for col in range(data.shape[0]):
                    threshed[col] = (data[col]>thresh*rms[col]).astype('int')
            else:
                threshed = (data > thresh*rms).astype('int')
            
            
            t_thresh = time.time() - t0
            if verbose: print('-threshold applied after %i seconds' % t_thresh )
            
            
            t_dbscan_start = time.time()
            
            #Obtain the core points
            if lowmem:
                
                corepts = convolution.convolve(threshed.astype('float32'),
                                               kernel=kernel_conv,
                                               meshsize=meshsize,
                                               Nthreads=Nthreads,
                                               lowmem=True,
                                               dtype='float32') 
                                                #Float to reduce artefacts
                
                for col in range(data.shape[0]):
                    corepts[col] = (corepts[col] >= mpts-fft_tol) * threshed[col]                     
            
            else: #If not lowmem
                corepts = convolution.convolve(threshed.astype('float32'),
                                               kernel=kernel_conv,lowmem=False)
                corepts = (corepts >= mpts-fft_tol) * threshed
            
            #Recast corepoints to integer (preserves memmap)
            corepts = corepts.astype('int')  
            
            
            ty = time.time() - t_dbscan_start
            if verbose: print('-corepoints obtained after %i seconds' % ty )
             
            #Obtain the area corresponding to the secondary points + core points
            if lowmem:
                secarea = convolution.convolve(corepts,
                                               kernel=kernel_conv,
                                               meshsize=meshsize,
                                               Nthreads=Nthreads,
                                               dtype='float32',
                                               lowmem=True)                                              
                for col in range(data.shape[0]):
                    secarea[col] = (secarea[col] >= 1-fft_tol) #Prevent negatives
                secarea = secarea.astype('int')
            else:
                secarea = convolution.convolve(corepts.astype('float32'),
                                               kernel=kernel_conv,
                                               lowmem=False)
                secarea = (secarea >= 1-fft_tol).astype('int')
                
            
            t_sec = time.time() - ty - t_dbscan_start
            if verbose: print('-secondary regions obtained after %i seconds.' % t_sec )
                    
    
            #Do the labeling on one processor
            labeled, Nlabels = label(secarea)   #Labeled is not memmap'd
            #slices_labeled = find_objects(labeled)
            slices = find_objects(labeled)
            
            
            t_labels = time.time()-t_sec-ty-t_dbscan_start
            if verbose: print('-Regions labeled after %i seconds.' % t_labels )
            
            #Label the clusters, preserving the memmap
            if lowmem:
                for row in range(corepts.shape[0]):
                    corepts[row] = corepts[row] * labeled[row]
            else:
                corepts[:, :] = corepts * labeled
            
            
            if Nlabels != 0: #If we have detections
                
                sources = []
                for i, slice_ in enumerate(slices):
                    sources.append( source.Source( i+1, slice_) )
                    
                if erode:
                    if verbose: print('-eroding...')
                    #Erode the clusters to represent the core point distribution only
                    segmap = erode_segmap(labeled, kernel_conv, 
                                          meshsize=meshsize, lowmem=lowmem,
                                          Nthreads=Nthreads, fft_tol=fft_tol)
                                          #Implicit memmap usage on lowmem
                    
                t_dbscan_finish = time.time() - t_dbscan_start
                
            else:  #If no detections
                if verbose: print('-no sources detected.')
                sources = []
                segmap = labeled #A zero array
                t_dbscan_finish = time.time() - t_dbscan_start
            
        finally:
            #Shut down the pool
            if pool is not None:
                pool.close()
                pool.join()
        
    return corepts, segmap, labeled, sources, t_dbscan_finish


#==============================================================================

class Clustered():
    '''
    Class to house DBSCAN detections.
    '''
    def __init__(self, corepoints, segmap, segmap_dilate, sources, t_dbscan=None):
        self.corepoints = corepoints
        self.segmap = segmap
        self.sources = sources
        self.segmap_dilate = segmap_dilate
        self.t_dbscan=t_dbscan
        self.Nobjs = len(sources)

#==============================================================================


def dbscan(data, eps, thresh, rms, ps, verbose=True, mask=None,
           mask_type='rms', sky=None, mpts=None, kappa=None, **kwargs):
    '''
    Run DBSCAN.
    
    Paramters
    ---------
    data: input data (2D numpy array)
    
    eps: Clustering radius (ps units)
    
    thresh: (SNR pixel threshold for clustering)
    
    ps: pixel scale 
    
    kappa: confidence parameter
    
    mpts: DBSCAN min points parameter. Overrides automatic derivation using kappa.
    
    mask_type: 'rms' or 'zeros': How should the mask be applied?
    
    Returns
    -------
    
    C: A Clustered object
    
    '''
    t0 = time.time()
    
    if sky is not None:
        data = data-sky
    
    if mask is not None:
        if mask_type == 'rms':
            data = masking.apply_mask(data, mask=mask, rms=rms, sky=sky)
        elif mask_type == 'zeros':
            data = masking.apply_mask(data, mask=mask, fillval=0)
        else:
            print('WARNING: mask_type not recongnised - ignoring mask...')
    
    eps = eps/ps
    
    if mpts is None:
        mpts = minpts.estimate_minpts(eps=eps, kappa=kappa, tmin=thresh, rms=1)
    
    if verbose: print('dbscan: performing clustering...')
    
    corepoints, segmap, segmap_dilate, sources, t_dbscan, = dbscan_conv(data, 
                                            thresh=thresh, eps=eps, mpts=mpts,     
                                             verbose=verbose, rms=rms, **kwargs)
    
    t1 = time.time() - t0
    
    if verbose: print('dbscan: finished after %i seconds.' % t1)
    return Clustered(corepoints, segmap, segmap_dilate, sources, t_dbscan=t_dbscan)
    
    
    
    
    
        


