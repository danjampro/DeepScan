#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:06:57 2017

@author: danjampro
"""

import numpy as np
from deepscan import skymap, masking, dbscan, remote_data
import matplotlib.pyplot as plt

#==============================================================================

print('Downloading data...')

#Load the remote data
data = remote_data.get('https://raw.github.com/danjampro/DeepScan/master/data/ngvs_small.fits.gz')

#Load in a source mask (from e.g. SExtractor, ProFound)
mask = remote_data.get('https://raw.github.com/danjampro/DeepScan/master/data/ngvs_small_mask.fits.gz')

print('Done.')

#==============================================================================

ps = 0.186
mzero = 30

#Measure the sky
bg, rms = skymap.skymap(data, mask=mask, wsize=50/ps)

#Apply bg subtraction 
data_ = data - bg

#Apply noise mask
masked = masking.apply_mask(data_, mask=mask, rms=rms)


#Run dbscan
clustered = dbscan.dbscan(eps=10, thresh=0.5, kappa=16, ps=ps, rms=rms, data=masked,
                          Nthreads=4) #Parallel by default
                                                          
                                                          
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(np.arcsinh(data), cmap='binary')
plt.subplot(1,3,2)
areas_ = clustered.segmap.astype('float'); areas_[areas_==0] = float(np.nan)
plt.imshow(areas_, cmap='hsv')
plt.subplot(1,3,3)
plt.imshow(mask, cmap='binary')
plt.tight_layout()
                         
plt.figure()
plt.imshow(np.arcsinh(data), cmap='binary')
plt.contour(clustered.segmap!=0, colors='lawngreen')   



#es = [s.get_ellipse_max(clustered.segmap) for s in clustered.sources]

                                            


