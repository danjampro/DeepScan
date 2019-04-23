#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:53:45 2018

@author: danjampro

Tools to deblend a detection image, avoiding fragmentation of LSB structure.
"""
import time
import numpy as np
from scipy.ndimage import label, find_objects
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import ks_2samp

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

#==============================================================================
#Deblend algorithm

def deblend(data, bmap, rms, contrast=0.5, minarea=5, alpha=1E-15,
            Nthresh=32, smooth=1, sky=0, verbose=True):
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
    
    Returns
    -------  
    2D array
        Deblended segmentation image.
    
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
    l0, uidmax = label(bmap>0, structure=structure)
    if verbose:
        print('-Initial number of segments: %i' % uidmax)
        
    #Find unique labels. This gets appended to following deblending.
    uids = list(range(1, uidmax))
        
    #Slice dict with keys of labels 
    slices = {_1+1:_2 for _1, _2 in enumerate(find_objects(l0))
                                                            if _2 is not None}                
    
    for uid0 in uids: #Loop over existing segments
                        
        #Apply slice to arrays to select labelled region
        slc = slices[uid0]
        l0_ = l0[slc]
        data_ = data[slc]
        rms_ = rms[slc]
                        
        #Calculate thresholds for the label
        if Nthresh == 'full':
            vs = np.unique(data_[l0_==uid0])  #Uses each pixel value (slow)
        else:
            quantiles = np.linspace(0, 1.0, Nthresh)
            vs = [np.quantile(data_[l0_==uid0], q) for q in quantiles] 
                    
        if vs[0]==vs[-1]: #One pixel, no deblending
            continue
        
        for v in vs[1:]:
            
            uids_ignore = []  #These do not get included in the new segmap
                
            #Label the new layer
            l1_, _ = label(data_>=v, structure=structure)
            l1_[l0_!=uid0] = 0
            uids1 = np.unique(l1_); uids1=uids1[uids1!=0]
            
            if l1_.max()<2: #At most one source detected (main branch)
                continue
        
            #This decides which pixels are used for the BG estimate
            databg = data_[(l0_==uid0) & (l1_==0)]
            
            #Identify main branch
            areas = [np.sum(l1_==_) for _ in uids1]
            uid_mainbranch = uids1[np.argmax(areas)]       
            
            #Apply attribute filtering
            for idx, uid_ in enumerate(uids1): #Loop over new objects
                
                #Main branch condition
                if uid_ == uid_mainbranch:
                    uids_ignore.append(uid_)
                    continue
                
                cond = l1_==uid_
                area = areas[idx]
                                               
                #Minumum area condition
                if area < minarea:
                    uids_ignore.append(uid_)
                    continue
                                
                #Mean flux condition (ignore if too faint)
                if data_[cond].mean()-v <= rms_[cond].mean() * contrast:
                    uids_ignore.append(uid_)
                    continue                    
                   
                #Statistical significance condition
                if not _KStest(databg, data_[cond], alpha=alpha):
                    uids_ignore.append(uid_)
                    continue
                                                                                     
            #Update the segmap
            uids_update = [_ for _ in uids1 if _ not in uids_ignore]
            cond = np.isin(l1_, uids_update, invert=False)
            l0[slc][cond] = uidmax + l1_[cond]
                    
            #Update uids
            uids.extend([uidmax + _ for _ in uids_update])
        
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
                
    if verbose:
        print('-Final number of segments: %i' % (len(uids)))
        print('deblend: finished after %i seconds.' % (time.time()-t0))
                                                            
    return l0

#==============================================================================
