#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 13:26:24 2017

@author: Dan
"""
import os, sys
from setuptools import setup 
from distutils.extension import Extension
import numpy as np

VERSION = '0.65'
#==============================================================================

if 'develop' in sys.argv:
    USE_CYTHON = True 
    ANNOTATE   = True 
    PROFILE    = True 
    
else:
    USE_CYTHON = False #Use Cython to generate c++ source files?
    ANNOTATE   = False #Annotate Cython files?
    PROFILE    = False #Profile Cython code?   

#==============================================================================

def no_cythonize(extensions, **_ignore):
    '''
    Adapt cython extensions to use ready made c/c++ source files.
    '''
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

if USE_CYTHON:
    try:
        from Cython.Build import cythonize
        ext_func = cythonize
    except ImportError:
        sys.stderr.write('Cython was not found!\n')
        sys.exit(-1)
else:
    ext_func = no_cythonize
    
#==============================================================================

extensions = [
        Extension('deepscan.cython.cy_deblend',
                  ['deepscan/cython/cy_deblend.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-std=c++11'],
                  extra_link_args=[],
                  language='c++'),
                  
        Extension('deepscan.cython.cy_skymap',
                   ['deepscan/cython/cy_skymap.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-std=c++11'],
                  extra_link_args=[],
                  language='c++'),
                  
        Extension('deepscan.cython.cy_makecat',
                  ['deepscan/cython/cy_makecat.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-std=c++11'],
                  extra_link_args=[],
                  language='c++'),
        
        Extension('deepscan.cython.cy_dbscan',
                  ['deepscan/cython/cy_dbscan.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-std=c++11'],
                  extra_link_args=[],
                  language='c++')
            ]
 
#==============================================================================
    
setup(name='deepscan',
      version=VERSION,
      description='DeepScan is a source extraction tool designed to identify \
extended low surface brightness features in large astronomical datasets.',
      url='https://github.com/danjampro/DeepScan',
      author='danjampro',
      author_email='danjampro@sky.com',
      license='GPL v3.0',
      packages=['deepscan'],
      package_data={'deepscan/cython':['*.pxd']},
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pandas',
          'astropy',
          'shapely'],
      zip_safe=False,
      ext_modules = ext_func(extensions, annotate=ANNOTATE,
                             compiler_directives={'profile':PROFILE})
      )

#==============================================================================
#==============================================================================
      
      
