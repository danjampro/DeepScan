#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:53:45 2018

@author: danjampro

Tools to deblend a detection image, avoiding fragmentation of LSB structure.
"""
import time
from copy import copy
import numpy as np
from scipy.ndimage import label, find_objects
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import ks_2samp, distributions
from scipy.ndimage.morphology import binary_dilation

from . import geometry, source

from .cython.cy_deblend import cy_Label, cy_DeblendSegment, cy_IndilateSegment

#==============================================================================
#Significance tests

def _KStest(fluxes0, fluxes1, alpha=1E-15):
    '''
    Perform a two-sided, two sample KS test at significance level alpha.
    
    Parameters
    ----------
    fluxes0 : 2D float array
        A sample of flux values.
        
    fluxes1 : 2D float array
        A sample of flux values.
        
    alpha : float
        Significance level. Higher values result in more deblended sources.
        
    Returns
    -------
    bool 
        True if significant.
    '''
    d, pval = ks_2samp(fluxes0, fluxes1)
    return (pval < alpha)


def _KStest_sorted(fluxes0_sorted, fluxes1, alpha=1E-15):
    '''
    Perform a two-sided, two sample KS test at significance level alpha.
    Using modified code from scipy.stats.ks_2samp.
    
    Parameters
    ----------
    fluxes0_sorted : 2D float array
        A sample of flux values sorted in ascending order.
        
    fluxes1 : 2D float array
        A sample of flux values.
        
    alpha : float
        Significance level. Higher values result in more deblended sources.
        
    Returns
    -------
    bool 
        True if significant.
    '''
    n1 = fluxes0_sorted.shape[0]
    n2 = fluxes1.shape[0]
    
    #Only sort fluxes1
    fluxes1.sort()
    
    #Calculate the statisitics
    data_all = np.concatenate([fluxes0_sorted,fluxes1])
    cdf0 = np.searchsorted(fluxes0_sorted,data_all,side='right')/(1.0*n1)
    cdf1 = (np.searchsorted(fluxes1,data_all,side='right'))/(1.0*n2)
    d = np.max(np.absolute(cdf0-cdf1))
    
    #Note: d absolute not signed distance
    en = np.sqrt(n1*n2/float(n1+n2))
    try:
        pval = distributions.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    except:
        pval = 1.0
    
    return (pval < alpha)

#==============================================================================
#Segment dilation
    
def dilate_segment(data, segmap, segID, parentID, makecopy=False, expand=5):
    '''
    Dilate a segment. Only pixels in the parent segment with brightness
    less than the minimum of the pre-dilated segment can be updated.
    
    Parameters
    ----------
    data : 2D float array
        The data.
        
    segmap : 2D int array
        The segmentation image.
        
    segID : int
        The segment ID of the segment to dilate.
        
    parentID : int
        The segment ID of the parent segment.
        
    makecopy : bool
        Make a copy of the segmentation image?
        
    expand : int
        The size of the dilation kernel.
    '''    
    if makecopy:
        segmap=segmap.copy()
      
    #Do the dilation
    kernel = geometry.unit_tophat(expand)  
    dilated = binary_dilation(segmap==segID, structure=kernel)
    
    #Update the segmap
    cond = (dilated>0) & (segmap==parentID) & (data<data[segmap==segID].min())
    segmap[cond] = segID
        
    return segmap

#==============================================================================
#Deblend algorithm - slow Python version

def _deblend(data, bmap, rms, contrast=0.5, minarea=5, alpha=1E-15,
             Nthresh=25, smooth=1, sky=0, verbose=True, expand=5):
    '''
    Deblend detected segments using a multi-threshold, bottom-up approach
    inspired by MTObjects (doi:10.1515/mathm-2016-0006).
        
    Thresholds are calculated using brightness quantiles for each segment.
    
    New labels are assigned if a child is statistically significant compared 
    to its parent. The child with the maximum area assumes the same label as
    its parent; this preserves extended LSB structure.
    
    Parameters
    ----------
    data : 2D float array
        The data array.
        
    bmap : 2D array
        Array marking detected pixels for deblending. Detected pixels have
        values>0.
        
    rms : 2D float array
        The sky RMS array.
        
    sky : 2D float array or float
        The sky value to subtract from the data, by default 0.
        
    contrast : float
        The minimum mean flux (defined in SNR units using rms) of a child
        node for it to be considered significant.
        
    minarea : int
        The minumim area (in pixels) for a significant child node.
        
    alpha : float
        Statistical significance level for the significance test.
        
    Nthresh : int
        Number of thresholds applied to each segment for deblending. These are
        different for each segment and are calculated as evenly spaced data
        quantiles. If Nthresh='full', uses every pixel value (slow!).
        
    smooth : float
        Width of the Gaussian smoothing kernel to be applied to sky subtracted
        data. smooth=None for no smoothing.
        
    verbose : bool
        Prints information if True.
        
    expand : int
        Radius of the dilation kernel used in dilate_segment in pixels. If 0
        or None, no dilation is performed.
    
    Returns
    -------  
    2D int array
        Deblended segmentation image. 
        
    List of source.Source objects.
        Sources corresponding to segments.
    ''' 
    if verbose:
        t0 = time.time()
        print('deblend: deblending...')
        
    #Apply BG subtraction and smoothing
    data = data-sky
    if smooth is not None:
        data = gaussian_filter(data, smooth)

    #Label contiguous regions in the bmap
    structure = np.ones((3,3),dtype='bool')
    segmap, uidmax = label(bmap>0, structure=structure)
    if verbose:
        print('-Initial number of segments: %i' % uidmax)
        
    #Unique segIDs. This gets appended to during deblending.
    uids = list(range(1, uidmax+1))
        
    #Slice dict with keys of labels 
    slices = {_1+1:_2 for _1, _2 in enumerate(find_objects(segmap))
                                                            if _2 is not None}
    #Dict containing parent IDs
    parents = {_:0 for _ in np.arange(1, uidmax+1)}  
    
    #Apply minimum area condition to uids (these do not get de-blended)
    areas0 = [np.sum(segmap[slices[uid]]==uid) for uid in uids]
    uids = [_1 for _1, _2 in zip(uids, areas0) if _2>=minarea]
    
    for uid0 in uids: #Loop over existing segments
                                
        #Apply slice to arrays to select labelled region
        slc = slices[uid0]
        segmap_ = segmap[slc]
        data_ = data[slc]
        rms_ = rms[slc]
                        
        #Calculate thresholds for the label
        if Nthresh == 'full':
            vs = np.unique(data_[segmap_==uid0])  #Uses each pixel value (slow)
        else:
            quantiles = np.linspace(0, 1.0, Nthresh)
            vs = [np.quantile(data_[segmap_==uid0], q) for q in quantiles] 
                                    
        #A binary map identifying the parent segment in the slice
        cond_parent = segmap_==uid0
                
        for v in vs[1:]:
            
            uids_ignore = []  #These do not get included in the new segmap
                
            #Label the new layer
            l1_, Nsrc = label( (data_>=v)&(cond_parent), structure=structure)
            
            if Nsrc<2: #At most one source detected (main branch)
                continue
                        
            #List of segIDs to test
            uids1 = np.arange(1, Nsrc+1)
        
            #This decides which pixels are used for the BG estimate
            databg = np.sort(data_[cond_parent & (l1_==0)])
            
            #Identify main branch
            areas = np.array([np.sum(l1_==_) for _ in uids1])
            uid_mainbranch = uids1[areas.argmax()]  
            
            #Apply minimum area condition
            uids1 = uids1[areas>=minarea]
            
            #Apply attribute filtering
            for idx, uid_ in enumerate(uids1): #Loop over new objects
                
                #Main branch condition
                if uid_ == uid_mainbranch:
                    uids_ignore.append(uid_)
                    continue
                                                
                #Mean flux condition (ignore if too faint)
                cond = l1_==uid_
                if data_[cond].mean()-v <= rms_[cond].mean() * contrast:
                    uids_ignore.append(uid_)
                    continue                    
                   
                #Statistical significance condition
                if not _KStest_sorted(databg, data_[cond], alpha=alpha):
                    uids_ignore.append(uid_)
                    continue
                                                                                     
            #Update the segmap
            uids_update = [_ for _ in uids1 if _ not in uids_ignore]
            cond = np.isin(l1_, uids_update, invert=False)
            segmap[slc][cond] = uidmax + l1_[cond]
            
            #Update parent condition
            cond_parent &= (~cond)
                    
            #Update uids & parents
            for uid_ in uids_update:
                uids.append(uidmax+uid_)
                parents[uidmax + uid_] = uid0
        
            #Update slices
            for idx, slc1 in enumerate(find_objects(l1_)):
                if (slc1 is not None) & (idx+1 in uids_update):
                    slc2 = (slice(slc1[0].start+slc[0].start,
                                  slc1[0].stop+slc[0].start),
                            slice(slc1[1].start+slc[1].start,
                                  slc1[1].stop+slc[1].start))
                    slices[idx+1+uidmax] = slc2
                            
            #Update uidmax
            if len(uids_update) != 0:
                uidmax = uidmax + np.max(uids_update)
                
    #Dilate child segments?           
    if (expand is not None) & (expand > 0):
        for segID, parentID in parents.items():
            if parentID > 0:
                slc = slices[parentID]                
                segmap[slc] = dilate_segment(data[slc], segmap[slc], segID,
                                        parentID,makecopy=False, expand=expand)
        
    #Create source list, including parentID        
    sources = [source.Source(_1,_2) for _1, _2 in slices.items()]
    for src in sources:
        src.series['parentID'] = parents[src.segID]
                
    if verbose:
        print('-Final number of segments: %i' % (len(uids)))
        print('deblend: finished after %i seconds.' % (time.time()-t0))
                                 
    #Return segmap and source list                                   
    return segmap, sources


#==============================================================================
#A much faster, cython version of _deblend
    
def deblend(data, bmap, rms, contrast=0.5, minarea=5, Nthresh=25, smooth=1,
            sky=0, verbose=True, expand=5, maxdepth=5):
    '''
    Deblend detected segments using a multi-threshold, bottom-up approach
    inspired by MTObjects (doi:10.1515/mathm-2016-0006).
        
    Thresholds are calculated using brightness quantiles for each segment.
    
    New labels are assigned if a child is statistically significant compared 
    to its parent. The child with the maximum area assumes the same label as
    its parent; this preserves extended LSB structure.
    
    Parameters
    ----------
    data : 2D float array
        The data array.
        
    bmap : 2D array
        Array marking detected pixels for deblending. Detected pixels have
        values>0.
        
    rms : 2D float array
        The sky RMS array.
        
    sky : 2D float array or float
        The sky value to subtract from the data, by default 0.
        
    contrast : float
        The minimum mean flux (defined in SNR units using rms) of a child
        node for it to be considered significant.
        
    minarea : int
        The minumim area (in pixels) for a significant child node.
                
    Nthresh : int
        Number of thresholds applied to each segment for deblending. These are
        different for each segment and are calculated as evenly spaced data
        quantiles. If Nthresh='full', uses every pixel value (slow!).
        
    smooth : float
        Width of the Gaussian smoothing kernel to be applied to sky subtracted
        data. smooth=None for no smoothing.
        
    verbose : bool
        Prints information if True.
        
    expand : int
        Radius of the dilation kernel used in dilate_segment in pixels. If 0
        or None, no dilation is performed.
        
    maxdepth : int
        Maximum number of nested layers.
    
    Returns
    -------  
    2D int array
        Deblended segmentation image. 
        
    List of source.Source objects.
        Sources corresponding to segments.
    ''' 
    if verbose:
        t0 = time.time()
        print('deblend: deblending...')
        
    data = gaussian_filter(data-sky, smooth)
        
    segmap, segments = cy_Label(data, bmap.astype(np.uint8))
    segIDmax = max([s.segID for s in segments])
    
    quantiles = np.linspace(0, 1.0, Nthresh)
    
    segments_at_depth = copy(segments)
    
    depth = 0; finished = False
    while not finished:
        
        finished = True #This needs to be proved wrong to terminate
        
        if depth > maxdepth: #Alternatively, terminate if this is True
            break
        
        segments_next_depth = []
        
        for segment in segments_at_depth:
            
            if segment.area < minarea:
                continue
            
            #Calculate thresholds
            slc = (slice(segment.ymin,segment.ymax),
                   slice(segment.xmin,segment.xmax))
                        
            cond = segmap[slc] == segment.segID
            data_ = data[slc][cond]
            threshes = np.quantile(data_, quantiles)  

            #Apply thresholding to segment            
            for thresh_ in threshes:
                segments_, segIDmax = cy_DeblendSegment(data, rms, segmap,
                                    thresh_, segment, minarea=minarea,
                                    segIDoffset=segIDmax+1, contrast=contrast) 
                                
                if len(segments_) != 0:
                    segments.extend(segments_)
                    segments_next_depth.extend(segments_)
                    finished = False
        
        depth += 1
        segments_at_depth = segments_next_depth
        
    #Do the dilations
    if (expand is not None) and (expand != 0):
        structure = geometry.unit_tophat(expand).astype(np.uint8)
        for segment in segments:         
            if segment.parentID != 0:
                cy_IndilateSegment(data, segmap, structure, segment)
    if verbose:
        print('-Final number of segments: %i' % (len(segments)))
        print('deblend: finished after %i seconds.' % (time.time()-t0))
        
    return segmap, segments
            
#==============================================================================
#==============================================================================

