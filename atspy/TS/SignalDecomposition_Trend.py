# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license


import pandas as pd
import numpy as np

from . import Time as tsti
from . import Perf as tsperf
from . import Plots as tsplot
from . import Utils as tsutil

import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

class cAbstractTrend:
    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = pd.DataFrame()
        self.mTrendPerf = tsperf.cPerf();
        self.mOutName = ""
        self.mFormula = None;
        self.mComplexity = None;

    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig).any() or np.isinf(sig).any() ):
            logger = tsutil.get_pyaf_logger();
            logger.error("TREND_RESIDUE_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.Internal_PyAF_Error("INVALID_COLUMN _FOR_TREND_RESIDUE ['"  + name + "'");
        pass


    def computePerf(self):
        if(self.mOptions.mDebug):
            self.check_not_nan(self.mTrendFrame[self.mOutName + '_residue'], self.mOutName + '_residue')
        # self.mTrendFrame.to_csv(self.mOutName + '_residue' + ".csv");

        self.mTrendFitPerf = tsperf.cPerf();
        self.mTrendForecastPerf = tsperf.cPerf();
        (lFrameFit, lFrameForecast, lFrameTest) = self.mSplit.cutFrame(self.mTrendFrame);
        self.mTrendFitPerf.compute(lFrameFit[self.mSignal] ,
                                   lFrameFit[self.mOutName], self.mOutName)
        self.mTrendForecastPerf.compute(lFrameForecast[self.mSignal] ,
                                        lFrameForecast[self.mOutName], self.mOutName)


class cConstantTrend(cAbstractTrend):
    def __init__(self):
        cAbstractTrend.__init__(self);
        self.mMean = 0.0
        self.mOutName = "ConstantTrend"
        self.mFormula = self.mOutName;    
        self.mComplexity = 0;
        
    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);

    def transformDataset(self, df):
        target = df[self.mSignal].values
        df[self.mOutName] = self.mMean * np.ones_like(df[self.mSignal]);
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;
    
    def fit(self):
        # real lag1
        lTrendEstimFrame = self.mSplit.getEstimPart(self.mTrendFrame);
        self.mMean = lTrendEstimFrame[self.mSignal].mean()
        target = self.mTrendFrame[self.mSignal]
        self.mTrendFrame[self.mOutName] = self.mMean * np.ones_like(target);
        self.mTrendFrame[self.mOutName + '_residue'] = target - self.mTrendFrame[self.mOutName]
        # self.mTrendFrame.to_csv("aaaa.csv")
        # print("cConstantTrend" , self.mMean);
        # self.mFormula = self.mOutName + "[" + str(self.mMean) + "]";    

    def compute(self):
        Y_pred = self.mMean
        return Y_pred

class cLag1Trend(cAbstractTrend):
    def __init__(self):
        cAbstractTrend.__init__(self);
        self.mDefaultValue = None
        self.mOutName = "Lag1Trend"
        self.mFormula = self.mOutName;
        self.mComplexity = 2;
        
    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);

    def replaceFirstMissingValue(self, df, series):
        # print(self.mDefaultValue, type(self.mDefaultValue));
        # Be explicit here .... some integer index does not work.
        df.loc[df.index[0] , series] = self.mDefaultValue;
        # print(df.head());
        
    def fit(self):
        # real lag1
        target = self.mTrendFrame[self.mSignal].values
        lEstim = self.mSplit.getEstimPart(self.mTrendFrame);
        self.mDefaultValue = lEstim[self.mSignal ].iloc[0]        
        self.mTrendFrame[self.mOutName] = self.mTrendFrame[self.mSignal].shift(1);
        # print(self.mTrendFrame[self.mSignal].shape , self.mTrendFrame[self.mOutName].shape)
        self.replaceFirstMissingValue(self.mTrendFrame, self.mOutName);
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values
        # print("cLag1Trend_FirstValue" , self.mDefaultValue);


    def transformDataset(self, df):
        target = df[self.mSignal].values
        df[self.mOutName] = df[self.mSignal].shift(1);
        self.replaceFirstMissingValue(df, self.mOutName);
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;

    def compute(self):
        Y_pred = self.mTrendFrame[self.mSignal].shift(1)
        return Y_pred


class cMovingAverageTrend(cAbstractTrend):
    def __init__(self, iWindow):
        cAbstractTrend.__init__(self);
        self.mOutName = "MovingAverage";
        self.mWindow = iWindow;
        self.mFormula = self.mOutName;
        self.mComplexity = 3;
        
    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);

    def fit(self):
        self.mOutName = "MovingAverage(" + str(self.mWindow) + ")";
        self.mFormula = self.mOutName;
        # real lag1
        target = self.mTrendFrame[self.mSignal].values
        self.mTrendFrame[self.mOutName] = self.mTrendFrame[self.mSignal].shift(1).rolling(self.mWindow).mean().fillna(method='bfill')
        mean = self.mSplit.getEstimPart(self.mTrendFrame)[self.mSignal].mean()
        self.mTrendFrame[self.mOutName].fillna(mean , inplace=True)
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values

    def transformDataset(self, df):
        target = df[self.mSignal].values
        df[self.mOutName] = df[self.mSignal].shift(1).rolling(self.mWindow).mean().fillna(method='bfill');
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;

    def compute(self):
        Y_pred = self.mTrendFrame[self.mSignal].shift(1)
        return Y_pred


class cMovingMedianTrend(cAbstractTrend):
    def __init__(self, iWindow):
        cAbstractTrend.__init__(self);
        self.mOutName = "MovingMedian";
        self.mWindow = iWindow;
        self.mFormula = self.mOutName;
        self.mComplexity = 3;
        
    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);

    def fit(self):
        self.mOutName = "MovingMedian(" + str(self.mWindow) + ")";
        self.mFormula = self.mOutName;
        # real lag1
        target = self.mTrendFrame[self.mSignal].values
        self.mTrendFrame[self.mOutName] = self.mTrendFrame[self.mSignal].shift(1).rolling(self.mWindow).median().fillna(method='bfill')
        mean = self.mSplit.getEstimPart(self.mTrendFrame)[self.mSignal].mean()
        self.mTrendFrame[self.mOutName].fillna(mean , inplace=True)
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values

    def transformDataset(self, df):
        target = df[self.mSignal].values
        df[self.mOutName] = df[self.mSignal].shift(1).rolling(self.mWindow).median().fillna(method='bfill');
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;

    def compute(self):
        Y_pred = self.mTrendFrame[self.mSignal].shift(1)
        return Y_pred


class cLinearTrend(cAbstractTrend):
    def __init__(self):
        cAbstractTrend.__init__(self);
        self.mTrendRidge = linear_model.Ridge()
        self.mOutName = "LinearTrend"
        self.mFormula = self.mOutName;
        self.mComplexity = 1;

    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);

    def fit(self):
        lTrendEstimFrame = self.mSplit.getEstimPart(self.mTrendFrame);
        est_target = lTrendEstimFrame[self.mSignal].values
        est_inputs = lTrendEstimFrame[[self.mTimeInfo.mNormalizedTimeColumn]].values
        self.mTrendRidge.fit(est_inputs, est_target)
        self.mTrendRidge.score(est_inputs, est_target)
        target = self.mTrendFrame[self.mSignal].values
        inputs = self.mTrendFrame[[self.mTimeInfo.mNormalizedTimeColumn]].values
        self.mTrendFrame[self.mOutName] = self.mTrendRidge.predict(inputs)
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values


    def transformDataset(self, df):
        target = df[self.mSignal].values
        inputs = df[[self.mTimeInfo.mNormalizedTimeColumn]].values
        df[self.mOutName] = self.mTrendRidge.predict(inputs)
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;

    def compute(self):
        lTimeAfterSomeSteps = self.mTimeInfo.nextTime(iSteps)
        lTimeAfterSomeStepsNormalized = self.mTimeInfo.normalizeTime(lTimeAfterSomeSteps)
        df = pd.DataFrame([lTimeAfterSomeStepsNormalized , lTimeAfterSomeStepsNormalized ** 2])
        Y_pred = self.mTrendRidge.predict(df.values)
        return Y_pred


class cPolyTrend(cAbstractTrend):
    def __init__(self):
        cAbstractTrend.__init__(self);
        self.mTrendRidge = linear_model.Ridge()
        self.mOutName = "PolyTrend"
        self.mFormula = self.mOutName
        self.mComplexity = 1;

    def addTrendInputVariables(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mOutName = self.mSignal + "_" + self.mOutName;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);
        self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn + "_^2"] = self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn] ** 2;    
        self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn + "_^3"] = self.mTrendFrame[self.mTimeInfo.mNormalizedTimeColumn] ** 3;    

    def fit(self):
        lTrendEstimFrame = self.mSplit.getEstimPart(self.mTrendFrame);
        est_target = lTrendEstimFrame[self.mSignal].values
        est_inputs = lTrendEstimFrame[
            [self.mTimeInfo.mNormalizedTimeColumn,
             self.mTimeInfo.mNormalizedTimeColumn + "_^2",
             self.mTimeInfo.mNormalizedTimeColumn + "_^3"]].values
        self.mTrendRidge.fit(est_inputs, est_target)
        self.mTrendRidge.score(est_inputs, est_target)
        target = self.mTrendFrame[self.mSignal].values
        inputs = self.mTrendFrame[
            [self.mTimeInfo.mNormalizedTimeColumn,
             self.mTimeInfo.mNormalizedTimeColumn + "_^2",
             self.mTimeInfo.mNormalizedTimeColumn + "_^3"]].values
        self.mTrendFrame[self.mOutName] = self.mTrendRidge.predict(inputs)
        self.mTrendFrame[self.mOutName + '_residue'] =  target - self.mTrendFrame[self.mOutName].values


    def transformDataset(self, df):
        df[self.mTimeInfo.mNormalizedTimeColumn + "_^2"] = df[self.mTimeInfo.mNormalizedTimeColumn] ** 2;    
        df[self.mTimeInfo.mNormalizedTimeColumn + "_^3"] = df[self.mTimeInfo.mNormalizedTimeColumn] ** 3;    
        target = df[self.mSignal].values
        inputs = df[
            [self.mTimeInfo.mNormalizedTimeColumn,
             self.mTimeInfo.mNormalizedTimeColumn + "_^2",
             self.mTimeInfo.mNormalizedTimeColumn + "_^3"]].values
        #print(inputs);
        pred = self.mTrendRidge.predict(inputs)
        df[self.mOutName] = pred;
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;


    def compute(self):
        lTimeAfterSomeSteps = self.mTimeInfo.nextTime(iSteps)
        lTimeAfterSomeStepsNormalized = self.mTimeInfo.normalizeTime(lTimeAfterSomeSteps)
        df = pd.DataFrame([lTimeAfterSomeStepsNormalized , lTimeAfterSomeStepsNormalized ** 2])
        Y_pred = self.mTrendRidge.predict(df.values)
        return Y_pred


class cTrendEstimator:
    
    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = pd.DataFrame()


    def needMovingTrend(self, df, i):
        N = df.shape[0];
        if(N < (12 * i)) :
            return False;
        return True;
        
    def defineTrends(self):

        self.mTrendList = [];
        
        if(self.mOptions.mActiveTrends['ConstantTrend']):
            self.mTrendList = [cConstantTrend()];
        
        if(self.mOptions.mActiveTrends['Lag1Trend']):
            self.mTrendList = self.mTrendList + [cLag1Trend()];

        N = self.mSignalFrame.shape[0];
        
        if(N > 1 and self.mOptions.mActiveTrends['LinearTrend']):
            self.mTrendList = self.mTrendList + [cLinearTrend()]

        if(N > 2 and self.mOptions.mActiveTrends['PolyTrend']):
            self.mTrendList = self.mTrendList + [cPolyTrend()]
                
        if(N > 2 and self.mOptions.mActiveTrends['MovingAverage']):
            for i in self.mOptions.mMovingAverageLengths:
                if(self.needMovingTrend(self.mSignalFrame , i)):
                    self.mTrendList = self.mTrendList + [cMovingAverageTrend(i)]

        if(N > 2 and self.mOptions.mActiveTrends['MovingMedian']):
            for i in self.mOptions.mMovingMedianLengths:
                if(self.needMovingTrend(self.mSignalFrame , i)):
                    self.mTrendList = self.mTrendList + [cMovingMedianTrend(i)]
        if(len(self.mTrendList) == 0):
            self.mTrendList = [cConstantTrend()];
            
        # logger = tsutil.get_pyaf_logger();
        # logger.info("ACTIVE_TRENDS" + str(self.mOptions.mActiveTrends));
        # logger.info("TRENDS" + str([tr.mOutName for tr in self.mTrendList]));


        
    def plotTrend(self):
        for trend in self.mTrendList:
            tsplot.decomp_plot(self.mTrendFrame, self.mTimeInfo.mNormalizedTimeColumn, self.mSignal, trend.mOutName , trend.mOutName + '_residue');
            

    def addTrendInputVariables(self):
        for trend in self.mTrendList:
            trend.addTrendInputVariables()
        pass

    def check_residue(self , sig, name):
#        print("check_not_nan "  + name);
#        print(sig);
        if(np.isnan(sig).any()):
            raise tsutil.Internal_PyAF_Error("Invalid residue '" + name + "'");
        pass

    def estimateTrends(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTrendFrame = pd.DataFrame()
        self.mTimeInfo.addVars(self.mTrendFrame);
        for trend in self.mTrendList:
            trend.mOptions = self.mOptions
            trend.fit();
            if(trend.mOptions.mDebugPerformance):
                trend.computePerf();
            self.mTrendFrame[trend.mOutName] = trend.mTrendFrame[trend.mOutName]
            self.mTrendFrame[trend.mOutName + "_residue"] = trend.mTrendFrame[trend.mOutName + "_residue"]
            if(self.mOptions.mDebug):
                self.check_residue(self.mTrendFrame[trend.mOutName + "_residue"].values,
                                   trend.mOutName + "_residue");
        pass

    def estimateTrend(self):
        self.defineTrends();
        for trend in self.mTrendList:
            trend.mSignalFrame = self.mSignalFrame;
            trend.mTimeInfo = self.mTimeInfo;            
            trend.mSplit = self.mSplit
        self.addTrendInputVariables();
        self.estimateTrends()
        
