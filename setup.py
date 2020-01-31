from setuptools import setup, Command
import os
import sys

setup(name='atspy',
      version='0.1',
      description='Automated Time Series in Python',
      url='https://github.com/firmai/automated-time-series',
      author='snowde',
      author_email='d.snow@firmai.org',
      license='MIT',
      packages=['atspy'],
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'numba',
          'datetime',

      ],
      zip_safe=False)
