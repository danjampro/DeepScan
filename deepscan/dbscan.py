#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:02:27 2017

@author: danjampro

"""

import os, tempfile, multiprocessing, shutil, time
import numpy as np
#from scipy.signal import fftconvolve
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_erosion
from astropy.convolution import Tophat2DKernel
#from functools import partial
from . import minpts, source, convolution, NTHREADS, BUFFSIZE
#from joblib import Parallel, delayed


#==============================================================================

"""
def perform_convolution(xmin, xmax, ymin, ymax, R, kernel, dshape):
    
    #Expand box
    xmax2 = xmax + R
    xmin2 = xmin - R
    ymin2 = ymin - R
    ymax2 = ymax + R
    
    #Look for boundary overlap
    xoverlap1 = np.max((0, -xmin2))           #Negative x boundary overlap
    xoverlap2 = np.max((0, xmax2-dshape[1]))  #Positive x boundary overlap
    yoverlap1 = np.max((0, -ymin2))           #Negative y boundary overlap
    yoverlap2 = np.max((0, ymax2-dshape[0]))  #Positive y boundary overlap
    
    #Crop
    xmax2 = int(np.min((xmax2, dshape[1])))
    ymax2 = int(np.min((ymax2, dshape[0])))
    xmin2 = int(np.max((xmin2, 0)))
    ymin2 = int(np.max((ymin2, 0)))
      
    cnv = fftconvolve(np.array(threshed[ymin2:ymax2,xmin2:xmax2]),
                                        kernel, mode='same').astype('int')
        
    conv[ymin:ymax, xmin:xmax] = cnv[R-yoverlap1:cnv.shape[0]-R+yoverlap2,
                                            R-xoverlap1:cnv.shape[1]-R+xoverlap2]
            
def perform_convolution2(xmin, xmax, ymin, ymax, R, kernel, dshape):
    
    #Expand box
    xmax2 = xmax + R
    xmin2 = xmin - R
    ymin2 = ymin - R
    ymax2 = ymax + R
    
    #Look for boundary overlap
    xoverlap1 = np.max((0, -xmin2))           #Negative x boundary overlap
    xoverlap2 = np.max((0, xmax2-dshape[1]))  #Positive x boundary overlap
    yoverlap1 = np.max((0, -ymin2))           #Negative y boundary overlap
    yoverlap2 = np.max((0, ymax2-dshape[0]))  #Positive y boundary overlap
    
    #Crop
    xmax2 = int(np.min((xmax2, dshape[1])))
    ymax2 = int(np.min((ymax2, dshape[0])))
    xmin2 = int(np.max((xmin2, 0)))
    ymin2 = int(np.max((ymin2, 0)))
      
    #Convolve and fill
    cnv = fftconvolve(np.array(conv[ymin2:ymax2,xmin2:xmax2]),
                                        kernel, mode='same').astype('int')
    conv2[ymin:ymax, xmin:xmax] = cnv[R-yoverlap1:cnv.shape[0]-R+yoverlap2,
                                            R-xoverlap1:cnv.shape[1]-R+xoverlap2]
"""   


def _erode_regions(sources, kernel):
    '''
    Perform binary errosion on slices.
    
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
def init(thresh_memmap, conv_memmap, conv_memmap2):
    global threshed, conv, labeled, conv2
    threshed = thresh_memmap
    conv = conv_memmap
    conv2 = conv_memmap2
"""    
    
def dbscan_conv(data, thresh, eps, mpts, Nthreads=NTHREADS,
                                         meshsize=None,
                                         thresh_type='SNR',
                                         rms=None,
                                         minCsize=5,
                                         erode=True,
                                         lowmem=False,
                                         verbose=True):
                                             
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
    kernel_conv = Tophat2DKernel(eps).array
    kernel_conv[kernel_conv!=0] = 1  #Require unit height
    
    #Create a tempory file for the memory map
    temppath = tempfile.mkdtemp()

    pool = None                                               
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
        
        
        if verbose: print('-convolving...')
        
        print('lowmem', lowmem)
        
        
        #Obtain the core points
        if lowmem:
            
            tx = time.time()
            
            corepts = convolution.convolve_large(threshed, kernel=kernel_conv,
                                                   meshsize=meshsize,
                                                   Nthreads=Nthreads,
                                                   dtype='int')
            
            ty = time.time() - tx 
            print(ty)
            
            print('modifying')
            for col in range(data.shape[0]):
                corepts[col] = (corepts[col] >= mpts+1) * threshed[col]
                
            tz = time.time() - ty - tx
            print(tz)
            
        else:
            corepts = convolution.convolve(threshed, kernel=kernel_conv)
            corepts = (corepts >= mpts+1) * threshed
        corepts = corepts.astype('int')  
        
        
        print('pass 1')
    
        
        #Obtain the area corresponding to the secondary points + core points
        if lowmem:
            secarea = convolution.convolve_large(corepts, kernel=kernel_conv,
                                                   meshsize=meshsize,
                                                   Nthreads=Nthreads,dtype='int')                                              
            for col in range(data.shape[0]):
                secarea[col] = (secarea[col] >= 1) #Prevent negatives
            secarea = secarea.astype('int')
        else:
            secarea = convolution.convolve(corepts, kernel=kernel_conv)
            secarea = (secarea >= 1).astype('int')
                

        if verbose: print('-labeling...')
        
        
        #Do the labeling on one processor
        labeled, Nlabels = label(secarea)
        slices_labeled = find_objects(labeled)
        
        '''
        #Select the clusters as core points - this was here (in wrong place before)
        corepoints *= threshed
        clusters = corepoints * labeled
        '''
        
        #Label the clusters
        clusters = corepts * labeled
        
        #Find unique cluster identifiers
        uids_ = np.unique(clusters)
        if uids_[0]==0:  #Remove 0 for cluster IDs
            uids_ = uids_[1:]
                
        #Remove clusters that are too small
        slices_clus = [s for s in find_objects(clusters) if s is not None]
        clens = [np.sum(clusters[s]==cid) for cid, s in zip(uids_, slices_clus)]
        
        uids = np.array([u for i, u in enumerate(uids_) if clens[i]>=minCsize])  
        #Delete the clusters than were eliminated
        if uids.size != uids_.size:          
            for i, s in enumerate(slices_clus):
                if (i+1) not in uids:
                    clusters[s][clusters[s]==(i+1)]=0
        Nclusters = uids.size
        
        if verbose: print('-cleaning...')
                            
        #Check if there are any non-continuities in the labeling
        if uids.size != 0:
            
            if (uids[-1] != Nlabels) or (Nlabels!=uids.size):
                
                #Select label_ids to keep
                label_ids = np.arange(1,Nlabels+1)
                label_slices_keep = np.ones(Nlabels, dtype='bool')
                for i, s in enumerate(slices_labeled):
                    if (i+1) not in uids:
                        
                        #Delete the labels with no clusters
                        labeled[s][labeled[s]==(i+1)]=0
                        label_slices_keep[i] = False
                        
                #Do the relabling
                ids_to_fill = (label_ids[:Nclusters])[~label_slices_keep[:Nclusters]]
                ids_to_replace = label_ids[Nclusters:][label_slices_keep[Nclusters:]]

                for i, id_fill in enumerate(ids_to_fill):
                    #Get slice of object to relable
                    slice_ = slices_labeled[ids_to_replace[i]-1]
                    #Do the relable
                    cond = labeled[slice_] == ids_to_replace[i]
                    labeled[slice_][cond] = id_fill
                    #Update the slices
                    slices_labeled[id_fill-1] = slices_labeled[ids_to_replace[i]-1]
                
                #Finish up
                clusters = corepts * labeled
                slices = slices_labeled[:Nclusters]
                
            else:
                slices = slices_labeled
        
            #Create source instances
            sources = []
            for i, slice_ in enumerate(slices):
                sources.append( source.Source( i+1, slice_) )
                
            if erode:
                if verbose: print('-eroding...')
                #Erode the clusters to represent the core point distribution only
                segmap = erode_segmap(labeled, sources, kernel_conv,
                                      directory=temppath, Nthreads=NTHREADS)
                    
        else:  #If no detections
            if verbose: print('-no sources detected.')
            sources = []
            segmap = None
    
    finally:
        
        #Remove the temporary directory
        shutil.rmtree(temppath)
        
        #Shut down the pool
        if pool is not None:
            pool.close()
            pool.join()
        
    return clusters, segmap, labeled, sources


#==============================================================================

class Clustered():
    '''
    Class to house DBSCAN detections.
    '''
    def __init__(self, clusters, segmap, segmap_dilate, sources):
        self.clusters = clusters
        self.segmap = segmap
        self.sources = sources
        self.segmap_dilate = segmap_dilate
        #self.mask = mask
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
    
    clusters, segmap, segmap_dilate, sources = dbscan_conv(data, thresh=thresh,
                                                           eps=eps, mpts=mpts,
                                             verbose=verbose, rms=rms, **kwargs)
    
    t1 = time.time() - t0
    
    if verbose: print('dbscan: finished after %i seconds.' % t1)
    return Clustered(clusters, segmap, segmap_dilate, sources)
    
    
    
    
    
        


