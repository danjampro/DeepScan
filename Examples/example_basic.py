#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:55:18 2019

@author: danjampro

A demonstration of the basic usage of DeepScan.
"""
from deepscan import remote_data
from deepscan.deepscan import DeepScan #DeepScan! (macro function)

#==============================================================================
#Load some data as a 2D numpy array

data = remote_data.get()

#==============================================================================
#Run the DeepScan macro function (uses default parameters unless specified)

result = DeepScan(data)  

#This is a Python dictionary containing a source catalogue ('cat'), 'segmap',
#the 'sky' and 'rms' estimates as well as a list of deepscan.source.Source
#('sources').

#The source catalogue is a pandas DataFrame. If you are unfamiliar then there
#is plenty of documentation online. If you want to save it as a .csv file then
#do >>>result['cat'].to_csv(filename).

#That's it.

#==============================================================================
#Make a plot

import numpy as np
import matplotlib.pyplot as plt
from deepscan import geometry

fig, ax = plt.subplots()

#Plot data
ax.imshow(np.arcsinh(data), cmap='binary', vmin=-0.5, vmax=3)

#Plot ellipses
for i in range(result['df'].shape[0]):
    s = result['df'].iloc[i]
    E = geometry.Ellipse(x0=s['xcen'], y0=s['ycen'], a=s['R50'], q=s['q'],
                         theta=s['theta'])
    E.draw(color='dodgerblue', linewidth=0.5, ax=ax)

#Format
ax.set_xlim(0, data.shape[1])
ax.set_ylim(data.shape[0], 0)

plt.show()
#et voila.

#==============================================================================
#==============================================================================





