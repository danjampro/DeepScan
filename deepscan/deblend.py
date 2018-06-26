#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:53:45 2018

@author: danjampro
"""

import tempfile
from . import utils

def watershed(data, tolerance=4, ext=2, mask=None):
    
    from rpy2 import robjects as ro
    
    data = data.copy()
    if mask is not None:
        data[mask==0] = 0
    
    with tempfile.NamedTemporaryFile() as tfile:
        utils.save_to_fits(data, tfile.name, overwrite=True)
        ro.r("require('EBImage')")
        ro.r("require('FITSio')")
        ro.r("image=readFITS('%s')$imDat" % tfile.name)
        ro.r("segim=EBImage::imageData(EBImage::watershed(image,tolerance=%.5f,ext=%.5f))" % (tolerance, ext))
        ro.r("segim=segim=matrix(as.integer(segim),dim(image)[1],dim(image)[2])")
        ro.r("writeFITSim(segim, '%s')" % tfile.name)
        
        return utils.read_fits(tfile.name)