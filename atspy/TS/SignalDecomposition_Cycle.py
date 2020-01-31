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

# for timing
import time


class cAbstractCycle:
    def __init__(self , trend):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = pd.DataFrame()
        self.mCycleFrame = pd.DataFrame()
        self.mTrend = trend;
        self.mTrend_residue_name = self.mTrend.mOutName + '_residue'
        self.mFormula = None;
        self.mComplexity = None;
    

    def getCycleResidueName(self):
        return self.getCycleName() + "_residue";


    def plot(self):
        tsplot.decomp_plot(self.mCycleFrame, self.mTimeInfo.mNormalizedTimeColumn,
                           self.mTrend_residue_name, self.getCycleName() , self.getCycleResidueName());


    def check_not_nan(self, sig , name):
        #print("check_not_nan");
        if(np.isnan(sig).any() or np.isinf(sig).any() ):
            logger = tsutil.get_pyaf_logger();
            logger.error("CYCLE_RESIDUE_WITH_NAN_IN_SIGNAL" + str(sig));
            raise tsutil.Internal_PyAF_Error("CYCLE_COLUMN _FOR_TREND_RESIDUE ['"  + name + "'");
        pass


    def computePerf(self):
        if(self.mOptions.mDebug):
            self.check_not_nan(self.mCycleFrame[self.getCycleResidueName()], self.getCycleResidueName())
        # self.mCycleFrame.to_csv(self.getCycleResidueName() + ".csv");
        self.mCycleFitPerf = tsperf.cPerf();
        self.mCycleForecastPerf = tsperf.cPerf();
        # self.mCycleFrame[[self.mTrend_residue_name, self.getCycleName()]].to_csv(self.getCycleName() + ".csv");
        (lFrameFit, lFrameForecast, lFrameTest) = self.mSplit.cutFrame(self.mCycleFrame);
        
        self.mCycleFitPerf.compute(
            lFrameFit[self.mTrend_residue_name], lFrameFit[self.getCycleName()], self.getCycleName())
        self.mCycleForecastPerf.compute(
            lFrameForecast[self.mTrend_residue_name], lFrameForecast[self.getCycleName()], self.getCycleName())
    

class cZeroCycle(cAbstractCycle):

    def __init__(self , trend):
        super().__init__(trend);
        self.mFormula = "NoCycle"
        self.mComplexity = 0;

    def getCycleName(self):
        return self.mTrend_residue_name + "_zeroCycle";


    def fit(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTimeInfo.addVars(self.mCycleFrame);
        self.mCycleFrame[self.mTrend_residue_name] = self.mTrendFrame[self.mTrend_residue_name]
        self.mCycleFrame[self.getCycleName()] = np.zeros_like(self.mTrendFrame[self.mTrend_residue_name])
        self.mCycleFrame[self.getCycleResidueName()] = self.mCycleFrame[self.mTrend_residue_name];
        self.mOutName = self.getCycleName()
        
    def transformDataset(self, df):
        target = df[self.mTrend_residue_name]
        df[self.getCycleName()] = np.zeros_like(df[self.mTrend_residue_name]);
        df[self.getCycleResidueName()] = target - df[self.getCycleName()].values        
        return df;

class cSeasonalPeriodic(cAbstractCycle):
    def __init__(self , trend, date_part):
        super().__init__(trend);
        self.mDatePart = date_part;
        self.mEncodedValueDict = {}
        self.mFormula = "Seasonal_" + self.mDatePart;
        self.mComplexity = 1;
        
        
    def getCycleName(self):
        return self.mTrend_residue_name + "_Seasonal_" + self.mDatePart;

    def hasEnoughData(self, iTimeMin, iTimeMax):
        lTimeDelta = iTimeMax - iTimeMin;
        lDays = lTimeDelta / np.timedelta64(1,'D');
        lSeconds = lTimeDelta / np.timedelta64(1,'s');
        if(self.mDatePart == "Hour"):
            return (lDays >= 10);
        if(self.mDatePart == "Minute"):
            lHours = lSeconds // 3600;
            return (lHours >= 10);
        if(self.mDatePart == "Second"):
            lMinutes = lSeconds // 60;
            return (lMinutes >= 10);
        if(self.mDatePart == "DayOfMonth"):
            lMonths = lDays // 30;
            return (lMonths >= 10);
        if(self.mDatePart == "DayOfWeek"):
            lWeeks = lDays // 7;
            return (lWeeks >= 10);
        if(self.mDatePart == "MonthOfYear"):
            lYears = lDays // 360;
            return (lYears >= 10);
        if(self.mDatePart == "WeekOfYear"):
            lYears = lDays // 360;
            return (lYears >= 10);
        
        return False;

    def fit(self):
        assert(self.mTimeInfo.isPhysicalTime());
        lHor = self.mTimeInfo.mHorizon;
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTimeInfo.addVars(self.mCycleFrame);
        lName = self.getCycleName();
        self.mCycleFrame[self.mTrend_residue_name] = self.mTrendFrame[self.mTrend_residue_name]
        self.mCycleFrame[lName] = self.mTrendFrame[self.mTime].apply(self.get_date_part);
        # we encode only using estimation
        lCycleFrameEstim = self.mSplit.getEstimPart(self.mCycleFrame);
        lTrendMeanEstim = lCycleFrameEstim[self.mTrend_residue_name].mean();
        lGroupBy = lCycleFrameEstim.groupby(by=[lName] , sort=False)[self.mTrend_residue_name].mean(); 
        self.mEncodedValueDict = lGroupBy.to_dict()
        self.mDefaultValue = lTrendMeanEstim;
        # print("cSeasonalPeriodic_DefaultValue" , self.getCycleName(), self.mDefaultValue);

        self.mCycleFrame[lName + '_enc'] = self.mCycleFrame[lName].apply(lambda x : self.mEncodedValueDict.get(x , self.mDefaultValue))
        self.mCycleFrame[lName + '_enc'].fillna(lTrendMeanEstim, inplace=True);
        self.mCycleFrame[self.getCycleResidueName()] = self.mCycleFrame[self.mTrend_residue_name] - self.mCycleFrame[lName + '_enc'];
        self.mCycleFrame[lName + '_NotEncoded'] = self.mCycleFrame[lName];
        self.mCycleFrame[lName] = self.mCycleFrame[lName + '_enc'];
        
        self.mOutName = self.getCycleName()
        #print("encoding '" + lName + "' " + str(self.mEncodedValueDict));


    @tsutil.cMemoize
    def get_date_part(self, x):
        lDatepartComputer = self.mTimeInfo.get_date_part_value_computer(self.mDatePart)
        return lDatepartComputer(x)
    

    @tsutil.cMemoize
    def get_date_part_encoding(self, x):
        lDatepartComputer = self.mTimeInfo.get_date_part_value_computer(self.mDatePart)
        dp = lDatepartComputer(x)
        return self.mEncodedValueDict.get(dp , self.mDefaultValue)

    def transformDataset(self, df):
        target = df[self.mTrend_residue_name]
        df[self.getCycleName()] = df[self.mTime].apply(self.get_date_part_encoding);
        df[self.getCycleResidueName()] = target - df[self.getCycleName()].values        
        return df;

class cBestCycleForTrend(cAbstractCycle):
    def __init__(self , trend, criterion):
        super().__init__(trend);
        self.mCycleFrame = pd.DataFrame()
        self.mCyclePerfDict = {}
        self.mBestCycleValueDict = {}
        self.mBestCycleLength = None
        self.mCriterion = criterion
        self.mComplexity = 2;
        self.mFormula = "BestCycle"
        
    def getCycleName(self):
        return self.mTrend_residue_name + "_bestCycle_by" + self.mCriterion;

    def dumpCyclePerfs(self):
        print(self.mCyclePerfDict);

    def computeBestCycle(self):
        # self.dumpCyclePerfs();
        lCycleFrameEstim = self.mSplit.getEstimPart(self.mCycleFrame);
        self.mDefaultValue = lCycleFrameEstim[self.mTrend_residue_name].mean();
        self.mBestCycleLength = None;
        lBestCycleIdx = None;
        lBestCriterion = None;
        if(self.mCyclePerfDict):
            for k in sorted(self.mCyclePerfDict.keys()):
                # smallest cycles are better
                if((lBestCriterion is None) or (self.mCyclePerfDict[k] < lBestCriterion)):
                    lBestCycleIdx = k;
                    lBestCriterion = self.mCyclePerfDict[k];
                    
            if(self.mOptions.mCycle_Criterion_Threshold is None or                 
                (self.mCyclePerfDict[lBestCycleIdx] < self.mOptions.mCycle_Criterion_Threshold)) :
                self.mBestCycleLength = lBestCycleIdx
        # print("BEST_CYCLE_PERF" , self.mTrend_residue_name, self.mBestCycleLength)


        self.transformDataset(self.mCycleFrame);
        pass


    def generate_cycles(self):
        self.mTimeInfo.addVars(self.mCycleFrame);
        self.mCycleFrame[self.mTrend_residue_name ] = self.mTrendFrame[self.mTrend_residue_name]
        lCycleFrameEstim = self.mSplit.getEstimPart(self.mCycleFrame);
        self.mDefaultValue = lCycleFrameEstim[self.mTrend_residue_name].mean();
        del lCycleFrameEstim;
        self.mCyclePerfDict = {}
        lMaxRobustCycle = self.mTrendFrame.shape[0]//12;
        # print("MAX_ROBUST_CYCLE_LENGTH", self.mTrendFrame.shape[0], lMaxRobustCycle);
        lCycleLengths = self.mOptions.mCycleLengths or range(2,lMaxRobustCycle + 1)
        lCycleFrame = pd.DataFrame();
        lCycleFrame[self.mTrend_residue_name ] = self.mTrendFrame[self.mTrend_residue_name]
        for i in lCycleLengths:
            if ((i > 1) and (i <= lMaxRobustCycle)):
                name_i = self.mTrend_residue_name + '_Cycle';
                lCycleFrame[name_i] = self.mCycleFrame[self.mTimeInfo.mRowNumberColumn] % i
                lCycleFrameEstim = self.mSplit.getEstimPart(lCycleFrame);
                lGroupBy = lCycleFrameEstim.groupby(by=[name_i] , sort=False)[self.mTrend_residue_name].mean();
                lEncodedValueDict = lGroupBy.to_dict()
                lCycleFrame[name_i + '_enc'] = lCycleFrame[name_i].apply(
                    lambda x : lEncodedValueDict.get(x , self.mDefaultValue))

                self.mBestCycleValueDict[i] = lEncodedValueDict;
                
                lPerf = tsperf.cPerf();
                # validate the cycles on the validation part
                lValidFrame = self.mSplit.getValidPart(lCycleFrame);
                lCritValue = lPerf.computeCriterion(lValidFrame[self.mTrend_residue_name],
                                                    lValidFrame[name_i + "_enc"],
                                                    self.mCriterion,
                                                    "Validation")
                self.mCyclePerfDict[i] = lCritValue;
                if(self.mOptions.mDebugCycles):
                    logger = tsutil.get_pyaf_logger();
                    logger.debug("CYCLE_INTERNAL_CRITERION " + name_i + " " + str(i) + \
                                 " " + self.mCriterion +" " + str(lCritValue))
        pass

    def fit(self):
        # print("cycle_fit" , self.mTrend_residue_name);
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.generate_cycles();
        self.computeBestCycle();
        self.mOutName = self.getCycleName()
        self.mFormula = "Cycle_None"
        if(self.mBestCycleLength is not None):
            self.mFormula = "Cycle" #  + str(self.mBestCycleLength);
        self.transformDataset(self.mCycleFrame);

    def transformDataset(self, df):
        if(self.mBestCycleLength is not None):
            lValueCol = df[self.mTimeInfo.mRowNumberColumn].apply(lambda x : x % self.mBestCycleLength);
            df['cycle_internal'] = lValueCol;
            # print("BEST_CYCLE" , self.mBestCycleLength)
            # print(self.mBestCycleValueDict);
            lDict = self.mBestCycleValueDict[self.mBestCycleLength];
            df[self.getCycleName()] = lValueCol.apply(lambda x : lDict.get(x , self.mDefaultValue));
        else:
            df[self.getCycleName()] = np.zeros_like(df[self.mTimeInfo.mRowNumberColumn]);            

        target = df[self.mTrend_residue_name]
        df[self.getCycleResidueName()] = target - df[self.getCycleName()].values
        if(self.mOptions.mDebug):
            self.check_not_nan(self.mCycleFrame[self.getCycleName()].values , self.getCycleName());

        return df;

class cCycleEstimator:
    
    def __init__(self):
        self.mTimeInfo = tsti.cTimeInfo()
        self.mTrendFrame = pd.DataFrame()
        self.mCycleFrame = pd.DataFrame()
        self.mCycleList = {}

    def addSeasonal(self, trend, seas_type, resolution):
        if(resolution >= self.mTimeInfo.mResolution):
            lSeasonal = cSeasonalPeriodic(trend, seas_type);
            if(self.mOptions.mActivePeriodics[lSeasonal.mFormula]):
                if(lSeasonal.hasEnoughData(self.mTimeInfo.mTimeMin, self.mTimeInfo.mTimeMax)):
                    self.mCycleList[trend] = self.mCycleList[trend] + [lSeasonal];
        pass
    
    def defineCycles(self):
        for trend in self.mTrendList:
            self.mCycleList[trend] = [];

            if(self.mOptions.mActivePeriodics['NoCycle']):
                self.mCycleList[trend] = [cZeroCycle(trend)];
            if(self.mOptions.mActivePeriodics['BestCycle']):
                self.mCycleList[trend] = self.mCycleList[trend] + [
                    cBestCycleForTrend(trend, self.mOptions.mCycle_Criterion)];
            if(self.mTimeInfo.isPhysicalTime()):
                # The order used here is mandatory. see filterSeasonals before changing this order.
                self.addSeasonal(trend, "MonthOfYear", tsti.cTimeInfo.sRES_MONTH);
                self.addSeasonal(trend, "WeekOfYear", tsti.cTimeInfo.sRES_DAY);
                self.addSeasonal(trend, "DayOfMonth", tsti.cTimeInfo.sRES_DAY);
                self.addSeasonal(trend, "DayOfWeek", tsti.cTimeInfo.sRES_DAY);
                self.addSeasonal(trend, "Hour", tsti.cTimeInfo.sRES_HOUR);
                self.addSeasonal(trend, "Minute", tsti.cTimeInfo.sRES_MINUTE);
                self.addSeasonal(trend, "Second", tsti.cTimeInfo.sRES_SECOND);

                
        for trend in self.mTrendList:
            if(len(self.mCycleList[trend]) == 0):
                self.mCycleList[trend] = [cZeroCycle(trend)];
            for cycle in self.mCycleList[trend]:
                cycle.mTrendFrame = self.mTrendFrame;
                cycle.mTimeInfo = self.mTimeInfo;
                cycle.mSplit = self.mSplit;
                cycle.mOptions = self.mOptions;
            
    def plotCycles(self):
        for trend in self.mTrendList:
            for cycle in self.mCycleList[trend]:
                cycle.plot()

    

    def dumpCyclePerf(self, cycle):
        if(self.mOptions.mDebugCycles):
            logger = tsutil.get_pyaf_logger();
            logger.debug("CYCLE_PERF_DETAIL_COUNT_FIT_FORECAST "  + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mCount) + " %.3f" % (cycle.mCycleForecastPerf.mCount));
            logger.debug("CYCLE_PERF_DETAIL_MAPE_FIT_FORECAST " + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mMAPE)+ " %.3f" % (cycle.mCycleForecastPerf.mMAPE));
            logger.debug("CYCLE_PERF_DETAIL_L2_FIT_FORECAST " + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mL2) +  " %.3f" % (cycle.mCycleForecastPerf.mL2));
            logger.debug("CYCLE_PERF_DETAIL_R2_FIT_FORECAST " + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mR2) +  " %.3f" % (cycle.mCycleForecastPerf.mR2));
            logger.debug("CYCLE_PERF_DETAIL_PEARSONR_FIT_FORECAST " + cycle.mOutName +
                  " %.3f" % (cycle.mCycleFitPerf.mPearsonR) +  " %.3f" % (cycle.mCycleForecastPerf.mPearsonR));


    def estimateCycles(self):
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTimeInfo.addVars(self.mCycleFrame);
        for trend in self.mTrendList:
            lTrend_residue_name = trend.mOutName + '_residue'
            self.mCycleFrame[lTrend_residue_name] = self.mTrendFrame[lTrend_residue_name]
            for cycle in self.mCycleList[trend]:
                start_time = time.time()
                cycle.fit();
                if(self.mOptions.mDebugPerformance):
                    cycle.computePerf();
                self.dumpCyclePerf(cycle)
                self.mCycleFrame[cycle.getCycleName()] = cycle.mCycleFrame[cycle.getCycleName()]
                self.mCycleFrame[cycle.getCycleResidueName()] = cycle.mCycleFrame[cycle.getCycleResidueName()]
                if(self.mOptions.mDebug):
                    cycle.check_not_nan(self.mCycleFrame[cycle.getCycleResidueName()].values ,
                                        cycle.getCycleResidueName())
                end_time = time.time()
                lTrainingTime = round(end_time - start_time , 2);
                if(self.mOptions.mDebugProfile):
                    logger = tsutil.get_pyaf_logger();
                    logger.info("CYCLE_TRAINING_TIME_IN_SECONDS '" + cycle.mOutName + "' " + str(lTrainingTime))
        pass


    def filterSeasonals(self):
        logger = tsutil.get_pyaf_logger();
        logger.debug("CYCLE_TRAINING_FILTER_SEASONALS_START")
        for trend in self.mTrendList:
            lPerfs = {}
            lTrend_residue_name = trend.mOutName + '_residue'
            lCycleList = []
            lSeasonals = []
            for cycle in self.mCycleList[trend]:
                if(isinstance(cycle , cSeasonalPeriodic)):
                    cycle.computePerf();
                    lPerfs[cycle.mOutName] = cycle.mCycleForecastPerf.getCriterionValue(self.mOptions.mCycle_Criterion)
                    lSeasonals = lSeasonals + [cycle]
                else:
                    lCycleList = lCycleList + [cycle]
            
            if(len(lSeasonals) == 0):
                return
            lBestCriterion = None
            lBestSeasonal = None
            for (k,cycle) in enumerate(lSeasonals):
                lCriterionValue = lPerfs[cycle.mOutName]
                if((lBestCriterion is None) or (lCriterionValue < (1.05 * lBestCriterion))):
                    lBestSeasonal = cycle
                    lBestCriterion = lCriterionValue;
            lCycleList = lCycleList + [lBestSeasonal]
            self.mCycleList[trend] = lCycleList
            logger.debug("CYCLE_TRAINING_FILTER_SEASONALS " + trend.mOutName + " " + lBestSeasonal.mOutName)
        logger.debug("CYCLE_TRAINING_FILTER_SEASONALS_END")
        pass

    def estimateAllCycles(self):
        self.defineCycles();
        self.estimateCycles()
        if(self.mOptions.mFilterSeasonals):
            self.filterSeasonals()
        
