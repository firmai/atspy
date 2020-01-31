# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np


class cModelControl:

    def __init__(self):
        self.mActiveTransformations = {};
        self.mActivePeriodics = {};
        self.mActiveTrends = {};
        self.mActiveAutoRegressions = {};
        self.mKnownTransformations = ['None', 'Difference', 'RelativeDifference',
                                      'Integration', 'BoxCox',
                                      'Quantization', 'Logit',
                                      'Fisher', 'Anscombe'];
        self.mKnownTrends = ['ConstantTrend', 
                             'Lag1Trend', 'LinearTrend', 'PolyTrend', 
                             'MovingAverage', 'MovingMedian'];
        self.mKnownPeriodics = ['NoCycle', 'BestCycle',
                                'Seasonal_MonthOfYear' ,
                                'Seasonal_Second' ,
                                'Seasonal_Minute' ,
                                'Seasonal_Hour' ,
                                'Seasonal_DayOfWeek' ,
                                'Seasonal_DayOfMonth',
                                'Seasonal_WeekOfYear'];

        # "AutoRegression" becomes a little bit confusing as croston does not use lags (???)
        # rather use wikipedia terminology :  https://en.wikipedia.org/wiki/Decomposition_of_time_series
        # AutoRegression => "irregular component"
        self.mKnownAutoRegressions = ['NoAR' , 'AR' , 'ARX' , 'SVR', 'SVRX', 'MLP' , 'LSTM' , 'XGB' , 'XGBX' , 'CROSTON'];
        # now , set he default models
        self.set_active_transformations(self.mKnownTransformations[0:4]);
        self.set_active_trends(self.mKnownTrends[0:4]);
        self.set_active_periodics(self.mKnownPeriodics);
        self.set_active_autoregressions(self.mKnownAutoRegressions[0:3]);
        
    def set_active_transformations(self, transformations):
        self.mActiveTransformations = {};
        for transformation in self.mKnownTransformations:
            if(transformation in transformations):
                self.mActiveTransformations[transformation] = True;
            else:
                self.mActiveTransformations[transformation] = False;
        if(True not in self.mActiveTransformations.values()):
            # default
            self.mActiveTransformations['None'] = True;
    
    def set_active_trends(self, trends):
        self.mActiveTrends = {};
        for trend in self.mKnownTrends:
            if(trend in trends):
                self.mActiveTrends[trend] = True;
            else:
                self.mActiveTrends[trend] = False;
        if(True not in self.mActiveTrends.values()):
            # default
            self.mActiveTrends['ConstantTrend'] = True;                
    
    def set_active_periodics(self, periodics):
        self.mActivePeriodics = {};
        for period in self.mKnownPeriodics:
            if(period in periodics):
                self.mActivePeriodics[period] = True;
            else:
                self.mActivePeriodics[period] = False;
        if(True not in self.mActivePeriodics.values()):
            # default
            self.mActivePeriodics['NoCycle'] = True;
                    
    def set_active_autoregressions(self, autoregs):
        self.mActiveAutoRegressions = {};
        for autoreg in self.mKnownAutoRegressions:
            if(autoreg in autoregs):
                self.mActiveAutoRegressions[autoreg] = True;
            else:
                self.mActiveAutoRegressions[autoreg] = False;                
        if(True not in self.mActiveAutoRegressions.values()):
            # default
            self.mActiveAutoRegressions['NoAR'] = True;

    def disable_all_transformations(self):
        self.set_active_transformations([]);
    
    def disable_all_trends(self):
        self.set_active_trends([]);
    
    def disable_all_periodics(self):
        self.set_active_periodics([]);
    
    def disable_all_autoregressions(self):
        self.set_active_autoregressions([]);
    
class cCrossValidationOptions:
    def __init__(self):
        self.mMethod = None;
        self.mNbFolds = 10

class cCrostonOptions:
    def __init__(self):
        # can be : "CROSTON" , "SBJ" , "SBA"
        self.mMethod = None;
        # alpha value or None to use optimal alpha based on RMSE
        self.mAlpha = 0.1
        # use "L2" by default, MAPE is not suitable (a lot of zeros in the signal) ?
        self.mAlphaCriterion = "L2"
        # minimum amount/percentage of zeros for a series to be intermittent
        self.mZeroRate = 0.1
        
class cSignalDecomposition_Options(cModelControl):
    
    def __init__(self):
        super().__init__();
        self.mParallelMode = True;
        self.mNbCores = 8;
        self.mEstimRatio = 0.8; # to be deprecated when cross validation is OK.
        self.mCustomSplit = None
        self.mAddPredictionIntervals = True
        self.enable_fast_mode();
        self.mTimeDeltaComputationMethod = "AVG"; # can be "AVG", "MODE", "USER"
        self.mUserTimeDelta = None;
        self.mBusinessDaysOnly = False;
        self.mMaxExogenousCategories = 5;
        self.mNoBoxCoxOrders = [];
        self.mBoxCoxOrders = [-2.0, -1.0 , 0.0,  2.0];
        self.mExtensiveBoxCoxOrders = [-2, -1, -0.5, -0.33 , -0.25 , 0.0, 2, 0.5, 0.33 , 0.25];
        self.mMaxFeatureForAutoreg = 1000;
        self.mModelSelection_Criterion = "MAPE";
        self.mCycle_Criterion = "L2";
        self.mCycle_Criterion_Threshold = None;
        self.mHierarchicalCombinationMethod = "BU";
        self.mForecastRectifier = None # can be "relu" to force positive forecast values
        self.mXGBOptions = None
        self.mCrossValidationOptions = cCrossValidationOptions()
        self.mCrostonOptions = cCrostonOptions()
        self.disableDebuggingOptions();

    def disableDebuggingOptions(self):
        self.mDebug = False;
        self.mDebugCycles = False;
        self.mDebugProfile = False;
        self.mDebugPerformance = False;
        
        
    def enable_slow_mode(self):
        self.mQuantiles = [5, 10, 20]; # quintiles, deciles, and vingtiles;)
        self.mMovingAverageLengths = [5, 7, 12, 24 , 30, 60];
        self.mMovingMedianLengths = [5, 7, 12, 24 , 30, 60];
        # PyAF does not detect complex seasonal patterns #73.
        # use unlimited cycle lengths in slow mode
        self.mCycleLengths = None;

        self.set_active_transformations(self.mKnownTransformations);
        self.set_active_trends(self.mKnownTrends);
        self.set_active_periodics(self.mKnownPeriodics);
        self.set_active_autoregressions(self.mKnownAutoRegressions);
        
        self.mMaxAROrder = 64;
        self.mFilterSeasonals = False
        # enable cross validation
        self.mCrossValidationOptions.mMethod = "TSCV";        

    def enable_fast_mode(self):
        self.mQuantiles = [5, 10, 20]; # quintiles, deciles, and vingtiles;)
        self.mMovingAverageLengths = [5, 7, 12, 24 , 30, 60];
        self.mMovingMedianLengths = [5, 7, 12, 24 , 30, 60];
        
        self.mCycleLengths = [5, 7, 12, 24 , 30, 60];

        self.mMaxAROrder = 64;
        self.mFilterSeasonals = True


    # Add a low-memory mode for Heroku #25
    def enable_low_memory_mode(self):
        self.mMaxAROrder = 7;
        self.set_active_transformations(['None']);
        self.mParallelMode = False;
        self.mFilterSeasonals = True
        
    '''
    Cannot yet build keras models in parallel/multiprocessing in some cases
    (tensorflow backend). theano seems OK.
    Possible solution : increase developer knowledge of keras !!
    '''
    def  canBuildKerasModel(self, iModel):
        try:
            import keras
            import keras
            from keras import callbacks
            from keras.models import Sequential
            from keras.layers import Dense, Dropout
            from keras.layers import LSTM
            lBackEnd = keras.backend.backend()
            if((lBackEnd == "tensorflow") and (self.mParallelMode)):
                return False;
            else:
                return True;
        except:
            return False;

