#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:22:58 2018

@author: danjampro

An example showing some of the basic DeepScan operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from deepscan import skymap, sextractor, dbscan, SB, geometry, remote_data

ps = 0.186 #Pixel scale [arcsec per pixel]
mzero = 30 #Magnitude zero point

data = remote_data.get('https://github.com/danjampro/DeepScan/tree/master/data/testimage1.fits.gz')

#from deepscan import utils
#data = utils.read_fits('../data/testimage1.fits')

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

mask = sextractor.get_mask(data, uiso=29, ps=ps, mzero=mzero, nfix=None)

plt.contour(mask!=0, colors='b', linewidths=0.3) #Mask contour

#==============================================================================
#Run the DBSCAN algorithm to produce a Clustered object (C). This class has
#many attributes such as the segmentation maps. 

C = dbscan.dbscan(data, eps=5, kappa=30, thresh=1, ps=ps,
                  verbose=True, mask=mask, sky=sky, rms=rms)

plt.contour(C.segmap_dilate!=0, colors='lawngreen', linewidths=0.5) #segmap contour

#==============================================================================
#Fit the source with a 1D Sersic profile. This is not particularly reliable (yet). 

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



