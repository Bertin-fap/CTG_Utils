#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
from os import path as OSPath

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = open(path.join(this_directory, 'requirements.txt'), encoding='utf-8').read().strip().split('\n') 

# This setup is suitable for "python setup.py develop".

setup(name='CTG_Utils',
      version='1.0.0',
      description='A toolbox for ctg statistics analysis',
      long_description=long_description,
      long_description_content_type='text/markdown',
      include_package_data = True,
      license = 'MIT',
      classifiers = [
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Information Analysis :: Visualization',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research'
        ],
      keywords = 'ffvelo,ctg',
      install_requires = install_requires,
      author= 'f bertin',
      author_email= 'francois.bertin7@wanadoo.fr',
      url= 'https://github.com/TickyWill/BiblioAnalysis_Utils',
      packages=find_packages(),
      )
