#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 13:26:24 2017

@author: Dan
"""

from setuptools import setup

setup(name='deepscan',
      version='0.1',
      description='DeepScan is a source extraction tool designed to identify very low surface brightness features in large astronomical data.',
      url='https://github.com/danjampro/DeepScan',
      author='danjampro',
      author_email='danjampro@sky.com',
      license='GPL v3.0',
      packages=['deepscan'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pandas',
          'astropy'],
      zip_safe=False)
