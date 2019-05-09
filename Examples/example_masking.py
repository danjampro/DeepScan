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
    
    Source masking using SExtractor with default settings (can be modified)
    
    DBSCAN source detection.
        
See the de-blending example for an approach that does not use a mask.
"""
import numpy as np
import matplotlib.pyplot as plt
from deepscan import skymap, sextractor, dbscan, SB, geometry, remote_data

#==============================================================================

ps = 0.186 #Pixel scale [arcsec per pixel]
mzero = 30 #Magnitude zero point

data = remote_data.get(1)  #Automatically deleted after download

plt.figure()  #Show the data using DeepScan's surface brightness transform
plt.imshow(SB.Counts2SB(abs(data), ps, mzero), cmap='binary_r', vmax=29,
           vmin=24)

#==============================================================================
#Measure the sky and its RMS in meshes that can be median filtered over.
#Niters is the number of DBSCAN masking iterations.

sky, rms = skymap.skymap(data, meshsize=200, medfiltsize=3, nits=3,
                         verbose=True)

#==============================================================================
#Make a SExtractor mask by running SExtractor and estimating isophotal radii
#of detections. 'uiso' is the isophotal surface brightness. 

#The Sersic index can be fixed with the "nfix" keyword. Higher values result
#in larger masks. nfix=4 is quite good. 

#The SExtractor settings can be specified here also. 
#See deepscan.sextractor.sextract.
#I will add some specific examples in the future.

mask = sextractor.get_mask(data, uiso=29, ps=ps, mzero=mzero, nfix=None)

plt.contour(mask!=0, colors='b', linewidths=0.3) #Mask contour

#==============================================================================
#Run the DBSCAN algorithm to produce a Clustered object (C). This class has
#many attributes such as the segmentation maps. 

#'eps' is the clustering radius in pixels. kappa is the confidence parameter
#determined from the thresh and the rms. 

#The automatic minpts derivation using kappa, rms and thresh can be overridden
#by specifying the mpts keyword argument.

C = dbscan.DBSCAN(data, eps=5/ps, kappa=30, thresh=1, verbose=True,
                  mask=mask, sky=sky, rms=rms, mpts=None)
#segmap contour
plt.contour(C.segmap!=0, colors='lawngreen', linewidths=0.5) 
plt.contour(C.segmap_dilate!=0, colors='deepskyblue', linewidths=0.5) 

#==============================================================================

#Use DeepScan's Ellipse class do draw the 1 Re ellipse
e0 = geometry.Ellipse(x0=500, y0=500, theta=3*np.pi/4, q=0.7, a=30/ps)
e0.draw(color='r', linewidth=1, zorder=1, linestyle='--')

#==============================================================================
#==============================================================================



