#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:02:27 2017

@author: danjampro

"""

import os, tempfile, multiprocessing, shutil, time
import numpy as np
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from functools import partial
from . import minpts, source, geometry, convolution, NTHREADS, BUFFSIZE

#==============================================================================
"""
def _dilate_regions(slices, kernel):
    '''
    Perform binary errosion on slices.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    for slc in slices:
        
        #print(slc)
        
        cutout = segmap_orig_[slc] 
        
        dilation = binary_dilation(cutout, kernel) 
        
        segmap_dilate_[slc] = dilation
      
        
def _init_dilate(segmap_in, segmap_out):
    '''
    Initializer for dilate_segmap process pool.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    global segmap_orig_, segmap_dilate_
    segmap_orig_ = segmap_in
    segmap_dilate_ = segmap_out
    
def dilate_corepts(corepts, kernel, directory=None, Nthreads=NTHREADS,
                  buffsize=BUFFSIZE):
    '''
    Perform binary dilation on the segmap.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    
    try:
        
        #Set up the output
        temppath = None
        if directory is None:
            temppath = tempfile.mkdtemp()
            fname = os.path.join(temppath, 'seg2.dat')
            segmap2 = np.memmap(fname,dtype='bool',mode='w+',shape=corepts.shape)
        else:
            segmap2 = np.memmap(os.path.join(directory, 'segmap2.dat'), dtype='bool',
                         shape=corepts.shape, mode='w+')
                
        #Do calculations for parallelisation
        slices = []
        chunksize = int(np.min((buffsize/corepts.dtype.itemsize, np.sqrt(corepts.size)/Nthreads)))
        for x in np.arange(0, corepts.shape[1], chunksize):
            for y in np.arange(0, corepts.shape[0], chunksize):
                slices.append( (slice(int(y),int(y+chunksize)),
                                slice(int(x),int(x+chunksize))) )
        n = int(len(slices)/Nthreads) + 1
        chunks = [slices[i:i + n] for i in range(0, len(slices), n)]
         
        #Create the pool
        pool = multiprocessing.Pool(processes=Nthreads,initializer=_init_dilate,
                                    initargs=(corepts, segmap2))
        pfunc = partial(_dilate_regions, kernel=kernel)
        pool.map(pfunc, chunks)
    
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if temppath is not None:
            shutil.rmtree(temppath)
            
    return segmap2
    

            

#==============================================================================

def _erode_regions(sources, kernel):
    '''
    Perform binary erosion on slices.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    for src in sources:
        
        cutout = segmap_orig[src.cslice] == src.label
        
        erosion = binary_erosion(cutout.astype('int'), kernel) * src.label
        
        lock_.acquire()
        segmap_eroded_[src.cslice] += erosion
        lock_.release()
      
        
def _init_erode(segmap_in, segmap_out, l):
    '''
    Initializer for erode_segmap process pool.
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    global segmap_orig, segmap_eroded_, lock_
    segmap_orig = segmap_in
    segmap_eroded_ = segmap_out
    lock_ = l
    
def erode_segmap(segmap, sources, kernel, directory, Nthreads=NTHREADS):
    '''
    Perform binary errosion on the segmap.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    #Do calculations for parallelisation
    n = int(len(sources)/Nthreads) + 1
    chunks = [sources[i:i + n] for i in range(0, len(sources), n)]
    
    #Pre-allocate a writeable shared memory map as a container for the
    #results of the parallel computation
    segmap2 = np.memmap(os.path.join(directory, 'segmap2.dat'), dtype='int',
                     shape=segmap.shape, mode='w+')
    
    #Create the pool
    l = multiprocessing.Lock()
    pool = multiprocessing.Pool(processes=Nthreads,initializer=_init_erode,
                                initargs=(segmap, segmap2, l))
    try:
        pool.starmap(_erode_regions, 
                     [[chunks[i], kernel] for i in range(len(chunks))])
    finally:
        pool.close()
        pool.join()
            
    return segmap2
"""
                    

def erode_segmap(segmap, kernel, directory, meshsize, lowmem=False, Nthreads=NTHREADS,
                 fft_tol=1E-5):
    
    conv = convolution.convolve_large((segmap==0).astype('float32'),kernel=kernel,
                                      Nthreads=Nthreads, meshsize=meshsize,
                                      dtype='float32')
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
        print('-threshold applied after %i seconds' % t_thresh )
        
        

        t_dbscan_start = time.time()
        
        #Obtain the core points
        if lowmem:
            
            corepts = convolution.convolve_large(threshed.astype('float32'), kernel=kernel_conv,
                                                   meshsize=meshsize,
                                                   Nthreads=Nthreads,
                                                   dtype='float32') #Float to reduce artefacts
            
            for col in range(data.shape[0]):
                corepts[col] = (corepts[col] >= mpts-fft_tol) * threshed[col]
                
            
        else:
            corepts = convolution.convolve(threshed.astype('float32'), kernel=kernel_conv)
            corepts = (corepts >= mpts-fft_tol) * threshed
        corepts = corepts.astype('int')  
        
        
        ty = time.time() - t_dbscan_start
        print('-corepoints obtained after %i seconds' % ty )
    
        
        #Obtain the area corresponding to the secondary points + core points
        if lowmem:
            secarea = convolution.convolve_large(corepts, kernel=kernel_conv,
                                                   meshsize=meshsize,
                                                   Nthreads=Nthreads,dtype='float32')                                              
            for col in range(data.shape[0]):
                secarea[col] = (secarea[col] >= 1-fft_tol) #Prevent negatives
            secarea = secarea.astype('int')
        else:
            secarea = convolution.convolve(corepts.astype('float32'), kernel=kernel_conv)
            secarea = (secarea >= 1-fft_tol).astype('int')
            
        
        t_sec = time.time() - ty - t_dbscan_start
        print('-secondary regions obtained after %i seconds.' % t_sec )
                

        #Do the labeling on one processor
        labeled, Nlabels = label(secarea)
        #slices_labeled = find_objects(labeled)
        slices = find_objects(labeled)
        
        
        t_labels = time.time()-t_sec-ty-t_dbscan_start
        print('-Regions labeled after %i seconds.' % t_labels )
        
        #Label the clusters
        corepts = corepts * labeled
        
        
        if Nlabels != 0:
            
            sources = []
            for i, slice_ in enumerate(slices):
                sources.append( source.Source( i+1, slice_) )
                
            if erode:
                if verbose: print('-eroding...')
                #Erode the clusters to represent the core point distribution only
                #segmap = erode_segmap(labeled, sources, geometry.unit_tophat(eps),
                #                      directory=temppath, Nthreads=Nthreads)
                segmap = erode_segmap(labeled, kernel_conv, temppath,
                                      meshsize, lowmem=lowmem, Nthreads=Nthreads,
                                      fft_tol=fft_tol)
                
            t_dbscan_finish = time.time() - t_dbscan_start
            
        else:  #If no detections
            if verbose: print('-no sources detected.')
            sources = []
            segmap = labeled
            t_dbscan_finish = time.time() - t_dbscan_start
        
    finally:
        
        #Remove the temporary directory
        shutil.rmtree(temppath)
        
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


def dbscan(data, eps, kappa, thresh, rms, ps, verbose=True, **kwargs):
    '''
    Run DBSCAN.
    
    Paramters
    ---------
    
    Returns
    -------
    
    '''
    t0 = time.time()
    
    eps = eps/ps
    
    mpts = minpts.estimate_minpts(eps=eps, kappa=kappa, tmin=thresh, rms=1)
    
    if verbose: print('dbscan: performing clustering...')
    
    corepoints, segmap, segmap_dilate, sources, t_dbscan, = dbscan_conv(data, 
                                            thresh=thresh, eps=eps, mpts=mpts,     
                                             verbose=verbose, rms=rms, **kwargs)
    
    t1 = time.time() - t0
    
    if verbose: print('dbscan: finished after %i seconds.' % t1)
    return Clustered(corepoints, segmap, segmap_dilate, sources, t_dbscan=t_dbscan)
    
    
    
    
    
        


