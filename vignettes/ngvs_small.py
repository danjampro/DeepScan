#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:06:57 2017

@author: danjampro
"""

import numpy as np
from deepscan import skymap, Fmask, dbscan, remote_data, minpts
import matplotlib.pyplot as plt

#==============================================================================

print('Downloading data...')

#Load the remote data
data = remote_data.get('https://raw.github.com/danjampro/DeepScan/master/data/ngvs_small.fits.gz')

#Load in a source mask (from e.g. SExtractor, ProFound)
mask = remote_data.get('https://raw.github.com/danjampro/DeepScan/master/data/ngvs_small_mask.fits.gz')

print('Done.')

#==============================================================================

ps = 0.187
mzero = 30

#Measure the sky
bg, rms = skymap.skymap(data, mask=mask, wsize=50/ps)

#Apply bg subtraction 
data_ = data - bg

#Apply noise mask
masked = Fmask.fmask(data_, mask=mask, rms=rms)

#Set DBSCAN parameters
eps = 10    #arcsec
thresh = 0  #SNR
kappa = 16 

#Estimate eta
mpts = minpts.estimate_minpts(eps=eps/ps, kappa=kappa, tmin=thresh, rms=1)

#Run dbscan
clusters, areas, sources = dbscan.dbscan_conv.dbscan_conv(eps=eps/ps,
                                            rms=rms,
                                            data=masked,
                                            thresh=thresh,
                                            mpts=mpts)  #Parallel by default
                                                          
                                                          
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(np.arcsinh(data), cmap='binary')
plt.subplot(1,3,2)
areas_ = areas.astype('float'); areas_[areas_==0] = float(np.nan)
plt.imshow(areas_, cmap='hsv')
plt.subplot(1,3,3)
plt.imshow(mask, cmap='binary')
plt.tight_layout()
                         
plt.figure()
plt.imshow(np.arcsinh(data), cmap='binary')
area_bin = (areas!=0).astype('int')
plt.contour(area_bin, colors='lawngreen')   

                         
                                            


