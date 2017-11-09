#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:06:57 2017

@author: danjampro
"""

import numpy as np
from deepscan import utils, skymap, Fmask, dbscan, remote_data, minpts
import matplotlib.pyplot as plt

#==============================================================================
#Online data retrieval
'''
url = 'https://raw.github.com/danjampro/DeepScan/master/data/ngvs_small.fits.gz'
fname_gz = os.path.join(os.getcwd(), 'ngvs_small.fits.gz')
fname = os.path.join(os.getcwd(), 'ngvs_small.fits')
urllib.request.urlretrieve(url, fname_gz)
with gzip.open(fname_gz, 'rb') as f_in, open(fname, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
'''

#==============================================================================
'''
#Load the remote data
data = remote_data.get('https://raw.github.com/danjampro/DeepScan/master/data/ngvs_small.fits.gz')

#Load in a source mask (from e.g. SExtractor, ProFound)
mask = remote_data.get('https://raw.github.com/danjampro/DeepScan/master/data/ngvs_small_mask.fits.gz')
'''
#==============================================================================

'''
fname1 = '/Users/danjampro/Dropbox/phd/codes/DeepScan/data/ngvs_small.fits'
fname2 = '/Users/danjampro/Dropbox/phd/codes/DeepScan/data/ngvs_small_mask.fits'
data = utils.read_fits(fname1)
mask = utils.read_fits(fname2)
'''

ps = 0.187
mzero = 30

#Measure the sky
bg, rms = skymap.skymap(data, mask=mask, wsize=50/ps)

#Apply bg subtraction 
data_ = data - bg

#Apply noise mask
masked = Fmask.fmask(data_, mask=mask, rms=rms)

#Run DBSCAN
eps = 10 #arcsec
thresh = 0 #SNR
kappa = 16 
mpts = minpts.estimate_minpts(eps=eps/ps, kappa=kappa, tmin=thresh, rms=1)

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
                                            


#Delete downloaded data
os.remove(fname_gz)
os.remove(fname)
