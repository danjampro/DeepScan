#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:06:57 2017

@author: danjampro
"""

import os
import urllib.request
import numpy as np
from deepscan import utils
import matplotlib.pyplot as plt

url = 'https://raw.github.com/danjampro/DeepScan/master/data/ngvs_small.fits'
fname = os.path.join(os.getcwd(), 'ngvs_small.fits')
urllib.request.urlretrieve(url, fname)
data = utils.read_fits(fname)


plt.figure(); plt.imshow(np.arcsinh(data), cmap='binary')