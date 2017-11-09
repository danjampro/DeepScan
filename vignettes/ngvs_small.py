#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:06:57 2017

@author: danjampro
"""

import os, gzip, shutil
import urllib.request
import numpy as np
from deepscan import utils
import matplotlib.pyplot as plt

#==============================================================================
#Online data retrieval

url = 'https://raw.github.com/danjampro/DeepScan/master/data/ngvs_small.fits.gz'
fname_gz = os.path.join(os.getcwd(), 'ngvs_small.fits.gz')
fname = os.path.join(os.getcwd(), 'ngvs_small.fits')
urllib.request.urlretrieve(url, fname_gz)
with gzip.open(fname_gz, 'rb') as f_in, open(fname, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)

data = utils.read_fits(fname)
#==============================================================================



plt.figure(); plt.imshow(np.arcsinh(data), cmap='binary')


#Delete downloaded data
os.remove(fname_gz)
os.remove(fname)