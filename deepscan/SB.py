#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Dan

-Conversions for Luminosities [counts] to Surface brightness and vice versa

-Total luminosity / magnitude from Sersic profile

"""

import numpy as np


def Mag2Counts(mag, mzero):
    """
    Convert magnitude to counts.
    
    Parameters
    ----------
    mag : magnitude [mag]
    
    mzero : magnitude zero point [counts]
    
    Returns
    -------
    counts : counts [counts]
    """
    exp = (mzero - mag)/2.5
    counts = 10**exp
    return counts
    
    
    
def Mag2SB(mag, ps, mzero):
    """
    Convert magnitude to surface brightness
    
    Parameters
    ----------
    mag : magnitude [mag]
    
    ps : pixel scale [arcsec per pixel]
    
    Returns
    -------
    SB : Surface Brightness [mag / arcsec**2]
    """
    counts = Mag2Counts(mag, mzero=mzero)
    counts_per_area = counts / ps ** 2
    SB = Counts2Mag(counts_per_area, mzero=mzero)
    return SB



def Counts2Mag(counts, mzero):
    """
    Convert counts to magnitude.
    
    Parameters
    ----------
    counts : counts [counts]
    
    mzero : magnitude zero point [counts]
    
    Returns
    -------
    mag : magnitude [mag]
    """
    mag = -2.5 * np.log10(counts) + mzero
    return mag



def Counts2SB(counts, ps, mzero):
    """
    Convert counts to surface brightness.
    
    Parameters
    ----------
    counts : counts [counts]
    
    ps : pixel scale [arcsec per pixel]
    
    Returns
    -------
    SB : Surface Brightness [mag / arcsec**2]
    """
    counts_per_area = counts / ps ** 2
    SB = Counts2Mag(counts_per_area, mzero=mzero)
    return SB



def SB2Counts(SB, ps, mzero):
    """
    Convert surface brightness to counts.
    
    Parameters
    ----------
    SB : Surface Brightness [mag / arcsec**2]
    
    ps : pixel scale [arcsec per pixel]
    
    Returns
    -------
    counts [counts]
    """
    #SB in mag/sqr arcsec
    counts_area = Mag2Counts(SB, mzero=mzero)
    counts = counts_area * ps**2
    return counts