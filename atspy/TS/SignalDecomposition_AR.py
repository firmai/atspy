# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

# from memory_profiler import profile

from . import Time as tsti
from . import Perf as tsperf
from . import Plots as tsplot
from . import Utils as tsutil


# for timing
import time

class cAbstractAR:
    def __init__(self , cycle_residue_name, iExogenousInfo = None):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mCycleFrame = pd.DataFrame()
        self.mARFrame = pd.DataFrame()        
        self.mCycleResidueName = cycle_residue_name
        self.mComplexity = None;
        self.mFormula = None;
        self.mTargetName = self.mCycleResidueName;
        self.mInputNames = None;
        self.mExogenousInfo = iExogenousInfo;

    def plot(self):
        tsplot.decomp_plot(self.mARFrame, self.mTimeInfo.mNormalizedTimeColumn,
                           self.mCycleResidueName, self.mOutName , self.mOutName + '_residue');

    def dumpCoefficients(self):
        pass

    def computePerf(self):
        self.mARFitPerf= tsperf.cPerf();
        self.mARForecastPerf= tsperf.cPerf();
        (lFrameFit, lFrameForecast, lFrameTest) = self.mSplit.cutFrame(self.mARFrame);
        self.mARFitPerf.compute(
            lFrameFit[self.mCycleResidueName], lFrameFit[self.mOutName], self.mOutName)
        self.mARForecastPerf.compute(
            lFrameForecast[self.mCycleResidueName], lFrameForecast[self.mOutName], self.mOutName)

    def shift_series(self, series, p, idefault):
        N = series.shape[0];
        lType = np.dtype(series);
        first_values = np.full((p), idefault, dtype=lType)
        new_values = np.hstack((first_values, series.values[0:N-p]));
        return new_values;

    def getDefaultValue(self, series):
        return self.mDefaultValues[series];

    def addLagForForecast(self, df, lag_df, series, p):
        name = series+'_Lag' + str(p);
        if(name not in self.mInputNames):
            return;
        lSeries = df[series];
        lShiftedSeries = self.shift_series(lSeries, p , self.mDefaultValues[series]); 
        lag_df[name] = lShiftedSeries;
        
    def generateLagsForForecast(self, df):
        lag_df = pd.DataFrame()
        lag_df[self.mCycleResidueName] = df[self.mCycleResidueName]
        for p in range(1, self.mNbLags + 1):
            # signal lags ... plain old AR model
            self.addLagForForecast(df, lag_df, self.mCycleResidueName, p);
        # Exogenous variables lags
        if(self.mExogenousInfo is not None):
            # print(self.mExogenousInfo.mEncodedExogenous);
            # print(df.columns);
            for p in range(1, self.mNbExogenousLags + 1):
                for ex in self.mExogenousInfo.mEncodedExogenous:
                    self.addLagForForecast(df, lag_df, ex, p);
        return lag_df;


class cZeroAR(cAbstractAR):
    def __init__(self , cycle_residue_name):
        super().__init__(cycle_residue_name, None)
        self.mOutName = self.mCycleResidueName +  '_NoAR'
        self.mNbLags = 0;
        self.mFormula = "NoAR";
        self.mComplexity = 0;
        
    def fit(self):
        series = self.mCycleResidueName; 
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        # self.mTimeInfo.addVars(self.mARFrame);
        # self.mARFrame[series] = self.mCycleFrame[series]
        self.mARFrame[self.mOutName] = self.mARFrame[series] * 0.0;
        self.mARFrame[self.mOutName + '_residue'] = self.mARFrame[series];
                

    def transformDataset(self, df, horizon_index = 1):
        series = self.mCycleResidueName; 
        df[self.mOutName] = 0.0;
        target = df[series].values
        df[self.mOutName + '_residue'] = target - df[self.mOutName].values        
        return df;



class cAutoRegressiveEstimator:
    def __init__(self):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mCycleFrame = pd.DataFrame()
        self.mARFrame = pd.DataFrame()
        self.mARList = {}
        self.mExogenousInfo = None;
        
    def plotAR(self):
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                for autoreg in self.mARList[cycle_residue]:
                    autoreg.plot(); 

    def is_not_constant(self, iSeries):
        lFirst = iSeries[0];
        for lValue in iSeries[1:]:
            if(lValue != lFirst):
                return True;
        return False;

    def shift_series(self, series, p):
        N = series.shape[0];
        lType = np.dtype(series);
        first_values = np.full((p), series.values[0], dtype=lType)
        new_values = np.hstack((first_values, series.values[0:N-p]));
        return new_values;

    def addLagForTraining(self, df, lag_df, series, autoreg, p):
        name = series+'_Lag' + str(p);
        if(name in lag_df.columns):
            autoreg.mInputNames.append(name);
            return lag_df;

        lSeries = df[series];
        lShiftedSeries = self.shift_series(lSeries, p);
        self.mDefaultValues[series] = lSeries.values[0];
        
        lShiftedEstim = self.mSplit.getEstimPart(lShiftedSeries);
        lAcceptable = self.is_not_constant(lShiftedEstim);
        if(lAcceptable):
            autoreg.mInputNames.append(name);
            lag_df[name] = lShiftedSeries;
            self.mLagOrigins[name] = series;
        return lag_df;

    def addLagsForTraining(self, df, cycle_residue, iNeedExogenous = False):
        logger = tsutil.get_pyaf_logger();
        add_lag_start_time = time.time()
        for autoreg in self.mARList[cycle_residue]:
            autoreg.mInputNames = [];
            P = autoreg.mNbLags;
            for p in range(1,P+1):
                # signal lags ... plain old AR model
                self.addLagForTraining(df, self.mARFrame, cycle_residue, autoreg, p);
            # Exogenous variables lags
            if(autoreg.mExogenousInfo is not None):
                P1 = P;
                lExogCount = len(autoreg.mExogenousInfo.mEncodedExogenous);
                lNbVars = P * lExogCount;
                if(lNbVars >= self.mOptions.mMaxFeatureForAutoreg):
                   P1 = self.mOptions.mMaxFeatureForAutoreg // lExogCount;
                autoreg.mNbExogenousLags = P1;
                for p in range(1,P1+1):
                    # print(autoreg.mExogenousInfo.mEncodedExogenous);
                    # print(df.columns);
                    for ex in autoreg.mExogenousInfo.mEncodedExogenous:
                        self.addLagForTraining(df, self.mARFrame, ex, autoreg, p);
            # print("AUTOREG_DETAIL" , P , len(autoreg.mInputNames));
            if(autoreg.mExogenousInfo is not None):
                assert((P + P*len(autoreg.mExogenousInfo.mEncodedExogenous)) >= len(autoreg.mInputNames));
            else:
                assert(P >= len(autoreg.mInputNames));
        if(self.mOptions.mDebugProfile):
            logger.info("LAG_TIME_IN_SECONDS " + self.mTimeInfo.mSignal + " " +
                  str(len(self.mARFrame.columns)) + " " +
                  str(time.time() - add_lag_start_time))

    # @profile
    def estimate_ar_models_for_cycle(self, cycle_residue):
        logger = tsutil.get_pyaf_logger();
        self.mARFrame = pd.DataFrame();
        self.mTimeInfo.addVars(self.mARFrame);
        self.mCycleFrame[cycle_residue] = self.mCycleFrame[cycle_residue].astype(np.float64)            
        self.mARFrame[cycle_residue] = self.mCycleFrame[cycle_residue].astype(np.float64)            

        self.mDefaultValues = {};
        self.mLagOrigins = {};

        if(self.mOptions.mDebugProfile):
            logger.info("AR_MODEL_ADD_LAGS_START '" +
                  cycle_residue + "' " + str(self.mCycleFrame.shape[0]) + " "
                  + str(self.mARFrame.shape[1]));

        self.addLagsForTraining(self.mCycleFrame, cycle_residue);

        if(self.mOptions.mDebugProfile):
            logger.info("AR_MODEL_ADD_LAGS_END '" +
                  cycle_residue + "' " + str(self.mCycleFrame.shape[0]) + " "
                  + str(self.mARFrame.shape[1]));

        # print(self.mARFrame.info());

        lCleanListOfArModels = [];
        for autoreg in self.mARList[cycle_residue]:
            if((autoreg.mFormula == "NoAR") or (len(autoreg.mInputNames) > 0)):
                lCleanListOfArModels.append(autoreg);
        self.mARList[cycle_residue] = lCleanListOfArModels;
        
        for autoreg in self.mARList[cycle_residue]:
            start_time = time.time()
            if(self.mOptions.mDebugProfile):
                logger.info("AR_MODEL_START_TRAINING_TIME '" +
                      cycle_residue + "' " + str(self.mCycleFrame.shape[0]) +
                      " " +  str(len(autoreg.mInputNames)) + " " + str(start_time));
            autoreg.mOptions = self.mOptions;
            autoreg.mCycleFrame = self.mCycleFrame;
            autoreg.mARFrame = self.mARFrame;
            autoreg.mTimeInfo = self.mTimeInfo;
            autoreg.mSplit = self.mSplit;
            autoreg.mLagOrigins = self.mLagOrigins;
            autoreg.mDefaultValues = self.mDefaultValues;
            autoreg.fit();
            if(self.mOptions.mDebugPerformance):
                autoreg.computePerf();
            end_time = time.time()
            lTrainingTime = round(end_time - start_time , 2);
            if(self.mOptions.mDebugProfile):
                logger.info("AR_MODEL_TRAINING_TIME_IN_SECONDS '" +
                      autoreg.mOutName + "' " + str(self.mCycleFrame.shape[0]) +
                      " " +  str(len(autoreg.mInputNames)) + " " + str(lTrainingTime));

    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig).any()):
            logger = tsutil.get_pyaf_logger();
            logger.error("CYCLE_RESIDUE_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.Internal_PyAF_Error("INVALID_COLUMN _FOR_CYCLE_RESIDUE ['"  + name + "'");
        pass


    
        
    # @profile
    def estimate(self):
        from . import Keras_Models as tskeras
        from . import Scikit_Models as tsscikit
        from . import Intermittent_Models as interm

        logger = tsutil.get_pyaf_logger();
        mARList = {}
        lNeedExogenous = False;
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                if(self.mOptions.mDebug):
                    self.check_not_nan(self.mCycleFrame[cycle_residue], cycle_residue)
                self.mARList[cycle_residue] = [];
                if(self.mOptions.mActiveAutoRegressions['NoAR']):
                    self.mARList[cycle_residue] = [ cZeroAR(cycle_residue)];
                lLags = self.mCycleFrame[cycle_residue].shape[0] // 4;
                if(lLags >= self.mOptions.mMaxAROrder):
                    lLags = self.mOptions.mMaxAROrder;
                if((self.mCycleFrame[cycle_residue].shape[0] > 12) and (self.mCycleFrame[cycle_residue].std() > 0.00001)):
                    if(self.mOptions.mActiveAutoRegressions['AR']):
                        lAR = tsscikit.cAutoRegressiveModel(cycle_residue, lLags);
                        self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lAR];
                    if(self.mOptions.mActiveAutoRegressions['ARX'] and (self.mExogenousInfo is not None)):
                        lARX = tsscikit.cAutoRegressiveModel(cycle_residue, lLags,
                                                             self.mExogenousInfo);
                        self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lARX];
                        lNeedExogenous = True;
                    if(self.mOptions.mActiveAutoRegressions['LSTM']):
                        if(self.mOptions.canBuildKerasModel('LSTM')):
                            lLSTM = tskeras.cLSTM_Model(cycle_residue, lLags,
                                                        self.mExogenousInfo);
                            self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lLSTM];
                        else:
                            logger.debug("SKIPPING_MODEL_WITH_KERAS '" + 'LSTM');
                        
                    if(self.mOptions.mActiveAutoRegressions['MLP']):
                        if(self.mOptions.canBuildKerasModel('MLP')):
                            lMLP = tskeras.cMLP_Model(cycle_residue, lLags,
                                                      self.mExogenousInfo);
                            self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lMLP];
                        else:
                            logger.debug("SKIPPING_MODEL_WITH_KERAS '" + 'MLP');
                        
                    if(self.mOptions.mActiveAutoRegressions['SVR']):
                        lSVR = tsscikit.cSVR_Model(cycle_residue, lLags);
                        self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lSVR];
                    if(self.mOptions.mActiveAutoRegressions['SVRX'] and (self.mExogenousInfo is not None)):
                        lSVRX = tsscikit.cSVR_Model(cycle_residue, lLags,
                                                       self.mExogenousInfo);
                        self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lSVRX];
                        lNeedExogenous = True;
                    if(self.mOptions.mActiveAutoRegressions['XGB']):
                        lXGB = tsscikit.cXGBoost_Model(cycle_residue, lLags)
                        self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lXGB];
                    if(self.mOptions.mActiveAutoRegressions['XGBX'] and (self.mExogenousInfo is not None)):
                        lXGBX = tsscikit.cXGBoost_Model(cycle_residue, lLags,
                                                       self.mExogenousInfo);
                        self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lXGBX];
                        lNeedExogenous = True;
                    if(self.mOptions.mActiveAutoRegressions['CROSTON']):
                        lIsSignalIntermittent = interm.is_signal_intermittent(self.mCycleFrame[cycle_residue] , self.mOptions)
                        if(lIsSignalIntermittent):
                            lCroston = interm.cCroston_Model(cycle_residue, lLags)
                            self.mARList[cycle_residue] = self.mARList[cycle_residue] + [lCroston];
                if(len(self.mARList[cycle_residue]) == 0):
                    self.mARList[cycle_residue] = [ cZeroAR(cycle_residue)];
                        

        if(lNeedExogenous):
            if(self.mOptions.mDebugProfile):
                logger.info("AR_MODEL_ADD_EXOGENOUS '" + str(self.mCycleFrame.shape[0]) +
                      " " + str(len(self.mExogenousInfo.mEncodedExogenous)));
            self.mCycleFrame = self.mExogenousInfo.transformDataset(self.mCycleFrame);
        
        for cycle_residue in self.mARList.keys():
            self.estimate_ar_models_for_cycle(cycle_residue);
            for autoreg in self.mARList[cycle_residue]:
                autoreg.mARFrame = pd.DataFrame();
            del self.mARFrame;
