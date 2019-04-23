#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:22:58 2018

@author: danjampro

An example showing some of the basic DeepScan operations.

The test data is a g-band cutout of the public NGVS data. The central LSB
source is synthetic. 

The basic outline demonstrated here is:
    
    sky / rms estimates: meshgrid + interpolation with iteritive DBSCAN masking
        
    DBSCAN source detection.
    
    De-blending of DBSCAN detections, avoiding fragmentation of LSB
    structure.
"""
import numpy as np
import matplotlib.pyplot as plt
from deepscan import skymap, dbscan, SB, remote_data, deblend

#==============================================================================
#Load the data

ps = 0.186 #Pixel scale [arcsec per pixel]
mzero = 30 #Magnitude zero point

print('Downloading data...')
data = remote_data.get(1)  #Automatically deleted after download

#==============================================================================
#Measure the sky 

sky, rms = skymap.skymap(data, meshsize=200, medfiltsize=3, nits=3,
                         verbose=True)

#==============================================================================
#Perform detection 

#Run the DBSCAN algorithm to produce a Clustered object (C). This class has
#many attributes such as the segmentation maps. 

#'eps' is the clustering radius in pixels. kappa is the confidence parameter
#determined from the thresh and the rms. 

C = dbscan.DBSCAN(data, eps=5, kappa=5, thresh=0.5, verbose=True, mask=None,
                  sky=sky, rms=rms)

#==============================================================================
#Perform deblending

bmap = C.segmap !=0 #Deblend the segmap produced by DBSCAN

segmap = deblend.deblend(data, bmap, rms, contrast=0.5, minarea=5, alpha=1E-15,
                         Nthresh=32, smooth=1, sky=sky, verbose=True)

#==============================================================================
#Summary plot - might take a few seconds
print('Plotting...')

plt.figure(figsize=(12,4)) 

plt.subplot(1,3,1) 
plt.imshow(SB.Counts2SB(abs(data), ps, mzero), cmap='binary_r', vmax=29,
           vmin=24)

plt.subplot(1,3,2) 
plt.imshow(SB.Counts2SB(abs(data), ps, mzero), cmap='binary_r', vmax=29,
           vmin=24)
plt.contour(segmap,levels = np.unique(segmap[segmap>0]), colors=('r',),
            linestyles=('-',), linewidths=(0.2,))  

plt.subplot(1,3,3)
labels = np.unique(segmap[segmap>0])
idx = labels[np.argmax([(segmap==_).sum() for _ in labels])]
data2 = np.zeros_like(data) * np.nan
data2[segmap==idx] = data[segmap==idx]
data3 = np.zeros_like(data) * np.nan
data3[segmap==0] = data[segmap==0]
plt.imshow(SB.Counts2SB(abs(data3), ps, mzero), vmax=29, vmin=24,
           cmap='binary_r')
plt.imshow(SB.Counts2SB(abs(data2), ps, mzero), vmax=29, vmin=24,
           cmap='viridis_r')

plt.tight_layout()

#==============================================================================
#==============================================================================




