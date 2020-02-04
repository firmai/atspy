from setuptools import setup, Command
import os
import sys

setup(name='atspy',
      version='0.0.9',
      description='Automated Time Series in Python',
      url='https://github.com/firmai/atspy',
      author='snowde',
      author_email='d.snow@firmai.org',
      license='MIT',
      packages=['atspy','atspy.TS'],
      install_requires=[
          'pandas',
          'scipy',
          'numba',
          'datetime',
          'pmdarima',
          'pydot',
          'dill',
          'pathos',
          'sqlalchemy',
          'matplotlib',
          'xgboost',
          'sklearn',
          'mxnet==1.4.1',
          'gluonts',
          'nbeats-pytorch',
          'seasonal',
          'tbats',
          'tsfresh',
          'python-dateutil==2.8.0',
          'numpy==1.17.4',

      ],
      zip_safe=False)
