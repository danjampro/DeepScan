#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:31:43 2017

@author: danjampro
"""

import numpy as np
from deepscan import utils, convolution, geometry, BUFFSIZE, NTHREADS
import matplotlib.pyplot as plt

data = utils.read_fits('/Users/danjampro/Dropbox/phd/data/VLSB_Ga.fits')

kernel = geometry.unit_tophat(10/0.187)
kernel /= kernel.sum()

meshsize = 3000#int( BUFFSIZE / data.dtype.itemsize) 
conv = convolution.convolve_large(data, kernel, meshsize=meshsize, Nthreads=NTHREADS)

plt.figure()
plt.imshow(np.arcsinh(data))
plt.figure()
plt.imshow(np.arcsinh(conv))
