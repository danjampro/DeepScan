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
    
    1D Sersic fitting.
    
I will get some more specific examples of each step uploaded in the future...
"""

import numpy as np
import matplotlib.pyplot as plt
from deepscan import skymap, sextractor, dbscan, SB, geometry, remote_data

ps = 0.186 #Pixel scale [arcsec per pixel]
mzero = 30 #Magnitude zero point

data = remote_data.get(1)  #Automatically deleted after download

plt.figure()  #Show the data using DeepScan's surface brightness transform
plt.imshow(SB.Counts2SB(abs(data), ps, mzero), cmap='binary_r', vmax=29, vmin=23)

#==============================================================================
#Measure the sky and its RMS in meshes that can be median filtered over.
#Niters is the number of DBSCAN masking iterations.

sky, rms = skymap.skymap(data, meshsize=200, medfiltsize=3, Niters=1,
                         verbose=True)

#==============================================================================
#Make a SExtractor mask by running SExtractor and estimating isophotal radii
#of detections. 'uiso' is the isophotal surface brightness. 

#The Sersic index can be fixed with the "nfix" keyword. Higher values result
#in larger masks. nfix=4 is quite good. 

#The SExtractor settings can be specified here also. See deepscan.sextractor.sextract.
#I will add some specific examples in the future.

mask = sextractor.get_mask(data, uiso=29, ps=ps, mzero=mzero, nfix=None)

plt.contour(mask!=0, colors='b', linewidths=0.3) #Mask contour

#==============================================================================
#Run the DBSCAN algorithm to produce a Clustered object (C). This class has
#many attributes such as the segmentation maps. 

#'eps' is the clustering radius in units specified by PS. i.e. if ps=1 then eps
#is in pixels. 'kappa' is the confidence parameter determined from the pixel 
#threshold 'thresh' and the rms. 

#The automatic minpts derivation using kappa, rms and thresh can be overridden
#by specifying the mpts keyword argument.

C = dbscan.dbscan(data, eps=5, kappa=30, thresh=1, ps=ps,
                  verbose=True, mask=mask, sky=sky, rms=rms, mpts=None)

plt.contour(C.segmap_dilate!=0, colors='lawngreen', linewidths=0.5) #segmap contour

#==============================================================================
#Fit the source with a 1D Sersic profile. This is not particularly reliable (yet). 
#It is strongly recommended to follow up with GALFIT, ProFit etc.

src = C.sources[0]
fit = src.fit_1Dsersic(data, segmap=C.segmap_dilate, ps=ps, mzero=mzero,
                       mask=mask, sky=sky, makeplots=False, dr=3)

#Make an Ellipse object to represent the fit
e = geometry.Ellipse(x0=fit['x0'], y0=fit['y0'], a=fit['re']/ps,
                     q=fit['q'])

e.draw(ax=plt.gca(), color='orange', linewidth=1) #Takes standard matplotlib kwargs

#Compare this to the actual synthetic source
e0 = geometry.Ellipse(x0=500, y0=500, theta=3*np.pi/4, q=0.7, a=30/ps)
e0.draw(color='r', linewidth=1, zorder=1)



