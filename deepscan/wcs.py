#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:48:52 2018

@author: danjampro
"""

import numpy as np

def arcsec_to_kpc(theta, d_mpc):
    '''
    Calculate the projected length of on-sky angle.
    
    Paramters
    ---------
    
    theta (float): On-sky angle [arcsec]
    
    d_mpc (float): Distance [Mpc]
    
    Returns
    -------
    
    kpc (float): Projected size [Kpc]
    '''
    return d_mpc*1E+3 * np.tan(theta * np.pi / 648000)


def kpc_to_arcsec(kpc, d_mpc):
    '''
    Calculate the projected length of on-sky angle.
    
    Paramters
    ---------
    
    kpc (float): Projected size [Kpc]
    
    d_mpc (float): Distance [Mpc]
    
    Returns
    -------
    
    theta (float): On-sky angle [arcsec]
    '''
    return 648000 / np.pi * np.arctan(kpc/(1000*d_mpc))