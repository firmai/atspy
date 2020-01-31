# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license



import pandas as pd
import numpy as np
import datetime

# this is a specialized missing data imputer for the time and signal
# it performs a simple interpolation (for the moment)

class cMissingDataImputer:

    def __init__(self):
        self.mOptions = None;


    def has_missing_data(self, iSeries):
        return iSeries.isnull().values.any()

    def interpolate_time_if_needed(self, iInputDS , iTime):
        if(not self.has_missing_data(iInputDS[iTime])):
            return iInputDS[iTime]
        
        type1 = np.dtype(iInputDS[iTime])
        if(type1.kind == 'M'):
            lMin = iInputDS[iTime].min()
            lDiffs = iInputDS[iTime] - lMin
            lDiffs = lDiffs.apply(lambda x : x.total_seconds())
            # print("TIME_MIN" , lMin)
            # print("TIME" , iInputDS[iTime].describe())
            # print("TIME_DIFFS" , lDiffs.describe())
            lTime = lDiffs.interpolate(method='linear', limit_direction='both', axis=0)
            lTime = lTime.apply(lambda x : lMin + datetime.timedelta(seconds=x))
            # print("TIME2" , lTime.describe())
            lTime = lTime.astype(type1)
            return lTime
        else:
            lTime = iInputDS[iTime].interpolate(method='linear', limit_direction='both', axis=0)
            lTime = lTime.astype(type1)
            return lTime
            
    def interpolate_signal_if_needed(self, iInputDS , iSignal):
        if(not self.has_missing_data(iInputDS[iSignal])):
            return iInputDS[iSignal]
        lSignal = iInputDS[iSignal].interpolate(method='linear', limit_direction='both', axis=0)
        lSignal = lSignal.astype(np.dtype(iInputDS[iSignal]))
        return lSignal
