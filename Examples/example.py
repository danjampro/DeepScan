#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:22:58 2018

@author: danjampro

An example showing some of the basic DeepScan operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from deepscan import skymap, sextractor, masking, dbscan, SB, geometry, remote_data

ps = 0.186 #Pixel scale [arcsec per pixel]
mzero = 30 #Magnitude zero point

data = remote_data.get('https://github.com/danjampro/DeepScan/tree/master/data/testimage1.fits.gz')

plt.figure()  #Show the data using DeepScan's surface brightness transform
plt.imshow(SB.Counts2SB(abs(data), ps, mzero), cmap='binary_r', vmax=29, vmin=23)

#==============================================================================
#Measure sky

sky, rms = skymap.skymap(data, meshsize=200, medfiltsize=3, Niters=1,
                         verbose=True)

#==============================================================================
#Make a SExtractor mask

aps = sextractor.get_ellipses(data, uiso=29, mzero=mzero, ps=ps)[0]
mask = masking.mask_ellipses(np.zeros_like(data), aps, fillval=1).astype('bool')

plt.contour(mask!=0, colors='r', linewidths=0.5) #Mask contour

#==============================================================================
#Do the clustering

C = dbscan.dbscan(data, eps=5, kappa=30, thresh=1, ps=ps,
                  verbose=True, mask=mask, sky=sky, rms=rms)

plt.contour(C.segmap_dilate!=0, colors='lawngreen', linewidths=0.5) #segmap contour

#==============================================================================
#Fit the source with a 1D Sersic profile

src = C.sources[0]
fit = src.fit_1Dsersic(data, segmap=C.segmap_dilate, ps=ps, mzero=mzero,
                       mask=mask, sky=sky, makeplots=False, dr=5)

#Make an Ellipse object to represent the fit
e = geometry.Ellipse(x0=fit['x0'], y0=fit['y0'], a=fit['re']/ps,
                     b=fit['re']/ps*fit['q'])

e.draw(color='deepskyblue', linewidth=1)





