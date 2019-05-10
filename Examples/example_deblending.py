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
    
    Creation of a source catalogue.
"""
import numpy as np
import matplotlib.pyplot as plt
from deepscan import skymap, dbscan, SB, remote_data, deblend, makecat,geometry

#==============================================================================
#Load the data

ps = 0.186 #Pixel scale [arcsec per pixel]
mzero = 30 #Magnitude zero point

print('Downloading data...')
data = remote_data.get()  #Automatically deleted after download

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
#Perform deblending, returning updated segmap and source list

bmap = C.segmap !=0 #Use the segmap produced by DBSCAN for deblending

segmap, sources = deblend.deblend(data, bmap, rms, contrast=0.5, minarea=5,
                                  alpha=1E-15, Nthresh=25, smooth=1, sky=sky,
                                  expand=5, verbose=True)

#==============================================================================
#Make catalogue using data, segmap and sources

cat = makecat.MakeCat(data, segmap, sources, sky=sky)
#This is a pandas.DataFrame

#==============================================================================
#Summary plot - might take a few seconds
print('Plotting...')

plt.figure(figsize=(12,4)) 

#Raw data
plt.subplot(1,3,1) 
plt.imshow(SB.Counts2SB(abs(data), ps, mzero), cmap='binary_r', vmax=28,
           vmin=23)

#Contour plot
plt.subplot(1,3,2) 
plt.imshow(SB.Counts2SB(abs(data), ps, mzero), cmap='binary_r', vmax=28,
           vmin=23)
plt.contour(segmap,levels = np.unique(segmap[segmap>0]), colors=('r',),
            linestyles=('-',), linewidths=(0.2,))  

#Ellipse plot (R50)
plt.subplot(1,3,3)
plt.imshow(SB.Counts2SB(abs(data), ps, mzero), cmap='binary_r', vmax=28,
           vmin=23)
for i in range(cat.shape[0]):
    s = cat.iloc[i]
    E = geometry.Ellipse(x0=s['xcen'], y0=s['ycen'], a=s['R50'], q=s['q'],
                         theta=s['theta'])
    E.draw(color='b', linewidth=0.5, ax=plt.gca())
plt.gca().set_xlim(0, data.shape[1])
plt.gca().set_ylim(data.shape[0], 0)

plt.tight_layout()

#==============================================================================
#==============================================================================




