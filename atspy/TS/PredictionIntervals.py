# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import SignalDecomposition as sigdec

from . import Perf as tsperf
from . import Utils as tsutil

class cPredictionIntervalsEstimator:
    
    def __init__(self):
        self.mModel = None;
        self.mSignalFrame = pd.DataFrame()
        self.mHorizon = -1;
        self.mFitPerformances = {}
        self.mForecastPerformances = {}
        self.mTestPerformances = {}

    def computePerformances(self):
        self.mTime = self.mModel.mTime;
        self.mSignal = self.mModel.mOriginalSignal;
        self.mHorizon = self.mModel.mTimeInfo.mHorizon;
        lTimeColumn = self.mTime;
        lSignalColumn = self.mSignal;
        lForecastColumn = str(self.mSignal) + "_Forecast";
        df = self.mModel.mTrend.mSignalFrame.reset_index();
        N = df.shape[0];
        (lOriginalFit, lOriginalForecast, lOriginalTest) = self.mModel.mTimeInfo.mSplit.cutFrame(df);
        df1 = df;
        for h in range(0 , self.mHorizon):
            df2 = None;
            df2 = self.mModel.forecastOneStepAhead(df1, horizon_index = h+1, perf_mode = True);
            df2 = df2.head(N);
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            (lFrameFit, lFrameForecast, lFrameTest) = self.mModel.mTimeInfo.mSplit.cutFrame(df2);
            self.mFitPerformances[lHorizonName] = tsperf.cPerf();
            self.mFitPerformances[lHorizonName].compute(lOriginalFit[lSignalColumn], lFrameFit[lForecastColumn], lHorizonName);
            self.mForecastPerformances[lHorizonName] = tsperf.cPerf();
            self.mForecastPerformances[lHorizonName].compute(lOriginalForecast[lSignalColumn], lFrameForecast[lForecastColumn], lHorizonName);
            self.mTestPerformances[lHorizonName] = tsperf.cPerf();
            if(lOriginalTest.shape[0] > 0):
                self.mTestPerformances[lHorizonName].compute(lOriginalTest[lSignalColumn], lFrameTest[lForecastColumn], lHorizonName);
            df1 = df2[[lTimeColumn , lForecastColumn]];
            df1.columns = [lTimeColumn , lSignalColumn]
        # self.dump_detailed();

    def dump_detailed(self):
        logger = tsutil.get_pyaf_logger();
        lForecastColumn = str(self.mSignal) + "_Forecast";
        for h in range(0 , self.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            hn = lHorizonName;
            logger.info("CONFIDENCE_INTERVAL_DUMP_FIT " +str(hn) + " " + str(self.mFitPerformances[hn].mL2) + " " + str(self.mFitPerformances[hn].mMAPE));
            logger.info("CONFIDENCE_INTERVAL_DUMP_FORECAST " +str(hn) + " " + str(self.mForecastPerformances[hn].mL2) + " " + str(self.mForecastPerformances[hn].mMAPE));
            logger.info("CONFIDENCE_INTERVAL_DUMP_TEST " +str(hn) + " " + str(self.mTestPerformances[hn].mL2) + " " + str(self.mTestPerformances[hn].mMAPE));


    def dump(self):
        logger = tsutil.get_pyaf_logger();
        lForecastColumn = str(self.mSignal) + "_Forecast";
        for h in range(0 , self.mHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            hn = lHorizonName;
            logger.info("CONFIDENCE_INTERVAL_DUMP_FORECAST " + str(hn) + " " + str(self.mForecastPerformances[hn].mL2));
            
