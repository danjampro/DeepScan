#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:20:25 2018

@author: danjampro
"""
import numpy as np
import matplotlib.pyplot as plt
from deepscan import SB, skymap, remote_data

ps = 0.186 #Pixel scale [arcsec per pixel]
mzero = 30 #Magnitude zero point

#==============================================================================
#Get the data
data = remote_data.get(1)  #Automatically deleted after download

#Add some sky
xx, yy = np.meshgrid(np.arange(0,data.shape[0],1.),
                     np.arange(0,data.shape[1],1.))
xx /= data.shape[1]; yy/= data.shape[1]
sky0 = SB.SB2Counts(27,ps,mzero) * (xx+yy)
data += sky0

#==============================================================================
#Measure the sky and its RMS in meshes that can be median filtered over.
#Niters is the number of DBSCAN masking iterations.

#Note that the mask used for the sky calculation can be returned using the
#"getmask" keyword argument

#"method" is the spline method. Can be 'cubic', 'linear' or 'nearest'.

#"tol" is the convergence tolerance for each mesh. Lower values mean larger
#masks and longer runtimes.

sky, rms, mask = skymap.skymap(data, meshsize=200, medfiltsize=3, nits=3,
                               verbose=True, getmask=True, method='cubic',
                               lowmem=False, tol=1.02)

#==============================================================================
#Makeplots 

plt.figure(figsize=(7,7))
plt.subplot(2,2,1) 
plt.imshow(SB.Counts2SB(abs(data), ps, mzero), cmap='binary_r',vmax=29,vmin=24)
plt.contour(mask!=0, colors='r', linewidths=0.5)
plt.xlabel('original data')
plt.subplot(2,2,2) 
plt.imshow(SB.Counts2SB(abs(data-sky), ps, mzero),cmap='binary_r',vmax=29,vmin=24)
plt.contour(mask!=0, colors='r', linewidths=0.5)
plt.xlabel('sky subtracted data')
plt.subplot(2,2,3)
plt.imshow(sky0) 
plt.xlabel('original sky')
plt.subplot(2,2,4)
plt.imshow(sky)
plt.xlabel('measured sky')
plt.tight_layout()

#==============================================================================
#==============================================================================
