#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:29:40 2019

@author: danjampro
"""

cdef class Segment:
    cdef public:
        Py_ssize_t xmin, xmax, ymin, ymax
        long area, segID, parentID
        double flux, fluxmin, fluxmax, rmssum