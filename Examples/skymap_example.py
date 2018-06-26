#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:33:06 2018

@author: danjampro

This is a short example of the skymap routine.
"""

import numpy as np
import matplotlib.pyplot as plt
from deepscan import SB, skymap

#==============================================================================
#Generate some data

ps = 0.2   #Pixel scale (e.g. arcsec per pixel)
mzero = 30 #Magnitude zero point

size = 500
data = np.ones((size, size), dtype='float') * SB.SB2Counts(25,ps,mzero)
data[0:int(size/2), 0:int(size/2)] = SB.SB2Counts(23,ps,mzero)
data[int(size/2):, int(size/2):] = SB.SB2Counts(23,ps,mzero)

xx, yy = np.meshgrid(np.arange(size), np.arange(size))
d2 = (xx-size/2)**2 + (yy-size/2)**2
data[d2 < (size/8)**2] = SB.SB2Counts(21,ps,mzero)

rms_actual = np.sqrt(data)
data = np.random.normal(data, rms_actual)

#==============================================================================

meshsize = 100   #BG mesh size in pixels
medfiltsize = 1  #Median filter size in mesh units
mask = None #Initial source mask
estimator_sky = np.median #Sky estimator
estimator_rms = skymap.rms_quantile #Sky RMS estimator
lowmem = True #Low memory mode? (memmaping)
tol = 1.03 #Convergence tolerance per mesh
Niters = 3 #Number of DBSCAN iterations
Nthreads = 1  #Number of threads
fillfrac = 0.3 #Minimum unmasked fraction in mesh before interpolation

sky, rms, mask = skymap.skymap(data, meshsize=meshsize, medfiltsize=medfiltsize,
                       mask=mask, estimator_sky=estimator_sky,
                       estimator_rms=estimator_rms, lowmem=lowmem,
                       tol=tol, Niters=Niters, Nthreads=Nthreads,
                       fillfrac=fillfrac, verbose=True, return_mask=True)

#==============================================================================

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(data)
plt.subplot(1,2,2)
plt.imshow(sky)
plt.contour(mask, colors='r', linewidths=0.3)
plt.tight_layout()

