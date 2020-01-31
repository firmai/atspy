# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import Utils as tsutil
from . import TimeSeries_Cutting as tscut


class cTimeInfo:
    # class data
    sRES_NONE = 0
    sRES_SECOND = 1
    sRES_MINUTE = 2
    sRES_HOUR = 3
    sRES_DAY = 4
    sRES_MONTH = 5
    sRES_YEAR = 6
    sDatePartComputer = {}
    sDatePartComputer["Second"] = lambda iTimeValue : iTimeValue.second
    sDatePartComputer["Minute"] = lambda iTimeValue : iTimeValue.minute
    sDatePartComputer["Hour"] = lambda iTimeValue : iTimeValue.hour
    sDatePartComputer["DayOfMonth"] = lambda iTimeValue : iTimeValue.day
    sDatePartComputer["DayOfWeek"] = lambda iTimeValue : iTimeValue.dayofweek
    sDatePartComputer["DayOfYear"] = lambda iTimeValue : iTimeValue.dayofyear
    sDatePartComputer["WeekOfYear"] = lambda iTimeValue : iTimeValue.weekofyear
    sDatePartComputer["MonthOfYear"] = lambda iTimeValue : iTimeValue.month        

    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mTimeMin = None;
        self.mTimeMax = None;
        self.mTimeMinMaxDiff = None;
        self.mTimeDelta = None;
        self.mHorizon = None;        
        self.mResolution = cTimeInfo.sRES_NONE
        self.mSplit = None

    def info(self):
        lStr2 = "TimeVariable='" + self.mTime +"'";
        lStr2 += " TimeMin=" + str(self.mTimeMin) +"";
        lStr2 += " TimeMax=" + str(self.mTimeMax) +"";
        lStr2 += " TimeDelta=" + str(self.mTimeDelta) +"";
        lStr2 += " Horizon=" + str(self.mHorizon) +"";
        return lStr2;


    def to_json(self):
        dict1 = {};
        dict1["TimeVariable"] =  self.mTime;
        dict1["TimeMinMax"] =  [str(self.mSignalFrame[self.mTime].min()) ,
                                str(self.mSignalFrame[self.mTime].max())];
        dict1["Horizon"] =  self.mHorizon;
        return dict1;

    def addVars(self, df):
        df[self.mRowNumberColumn] = self.mSignalFrame[self.mRowNumberColumn]
        df[self.mTime] = self.mSignalFrame[self.mTime]
        df[self.mNormalizedTimeColumn] = self.mSignalFrame[self.mNormalizedTimeColumn]
        df[self.mSignal] = self.mSignalFrame[self.mSignal]
        df[self.mOriginalSignal] = self.mSignalFrame[self.mOriginalSignal]

    def get_time_dtype(self):
        # print(self.mTimeMax, type(self.mTimeMax))
        lType = np.dtype(self.mTimeMax);
        return lType;

    def cast_to_time_dtype(self, iTimeValue):
        lType1 = self.get_time_dtype();
        lTimeValue = np.array([iTimeValue]).astype(lType1)[0];
        return lTimeValue;

    def checkDateAndSignalTypesForNewDataset(self, df):
        if(self.mTimeMax is not None):
            lType1 = self.get_time_dtype();
            lType2 = np.dtype(df[self.mTime]);
            if(lType1.kind != lType2.kind):
                raise tsutil.PyAF_Error('Incompatible Time Column Type expected=' + str(lType1) + ' got: ' + str(lType2) + "'");
                pass
        

    def transformDataset(self, df):
        self.checkDateAndSignalTypesForNewDataset(df);
        # new row
        lLastRow = df.tail(1).copy();
        lLastRow[self.mTime] = self.nextTime(df, 1);
        lLastRow[self.mSignal] = np.nan;
        # print(lLastRow.columns ,  df.columns)
        assert(str(lLastRow.columns) == str(df.columns))
        df = df.append(lLastRow, ignore_index=True, verify_integrity = True, sort=False);        
        df[self.mRowNumberColumn] = np.arange(0, df.shape[0]);
        df[self.mNormalizedTimeColumn] = self.compute_normalize_date_column(df[self.mTime])
        # print(df.tail());
        return df;


    def isPhysicalTime(self):
        type1 = np.dtype(self.mSignalFrame[self.mTime])
        return (type1.kind == 'M');


    def get_date_part_value_computer(self , iDatePart):
        return cTimeInfo.sDatePartComputer[iDatePart];
    
    def analyzeSeasonals(self):
        if(not self.isPhysicalTime()):
            return;
        lEstim = self.mSplit.getEstimPart(self.mSignalFrame);
        lEstimSecond = lEstim[self.mTime].apply(self.get_date_part_value_computer("Second"));
        if(lEstimSecond.nunique() > 1.0):
            self.mResolution = cTimeInfo.sRES_SECOND;
            return;
        lEstimMinute = lEstim[self.mTime].apply(self.get_date_part_value_computer("Minute"));
        if(lEstimMinute.nunique() > 1.0):
            self.mResolution =  cTimeInfo.sRES_MINUTE;
            return;
        lEstimHour = lEstim[self.mTime].apply(self.get_date_part_value_computer("Hour"));
        if(lEstimHour.nunique() > 1.0):
            self.mResolution =  cTimeInfo.sRES_HOUR;
            return;
        lEstimDayOfMonth = lEstim[self.mTime].apply(self.get_date_part_value_computer("DayOfMonth"));
        if(lEstimDayOfMonth.nunique() > 1.0):
            self.mResolution =  cTimeInfo.sRES_DAY;
            return;
        lEstimMonth = lEstim[self.mTime].apply(self.get_date_part_value_computer("MonthOfYear"));
        if(lEstimMonth.nunique() > 1.0):
            self.mResolution =  cTimeInfo.sRES_MONTH;
            return;
        self.mResolution =  cTimeInfo.sRES_YEAR;


    def checkDateAndSignalTypes(self):
        # print(self.mSignalFrame.info());
        type1 = np.dtype(self.mSignalFrame[self.mTime])
        if(type1.kind == 'O'):
            raise tsutil.PyAF_Error('Invalid Time Column Type ' + self.mTime + '[' + str(type1) + ']');
        type2 = np.dtype(self.mSignalFrame[self.mSignal])
        if(type2.kind == 'O'):
            raise tsutil.PyAF_Error('Invalid Signal Column Type ' + self.mSignal);
        


    def adaptTimeDeltaToTimeResolution(self):
        if(not self.isPhysicalTime()):
            return;
        if(cTimeInfo.sRES_SECOND == self.mResolution):
            self.mTimeDelta = pd.DateOffset(seconds=round(self.mTimeDelta / np.timedelta64(1,'s')))
            return;
        if(cTimeInfo.sRES_MINUTE == self.mResolution):
            self.mTimeDelta = pd.DateOffset(minutes=round(self.mTimeDelta / np.timedelta64(1,'m')))
            return;
        if(cTimeInfo.sRES_HOUR == self.mResolution):
            self.mTimeDelta = pd.DateOffset(hours=round(self.mTimeDelta / np.timedelta64(1,'h')))
            return;
        if(cTimeInfo.sRES_DAY == self.mResolution):
            self.mTimeDelta = pd.DateOffset(days=round(self.mTimeDelta / np.timedelta64(1,'D')))
            return;
        if(cTimeInfo.sRES_MONTH == self.mResolution):
            self.mTimeDelta = pd.DateOffset(months=round(self.mTimeDelta // np.timedelta64(30,'D')))
            return;
        if(cTimeInfo.sRES_YEAR == self.mResolution):
            self.mTimeDelta = pd.DateOffset(months=round(self.mTimeDelta // np.timedelta64(365,'D')))
            return;
        pass
    
    def get_lags_for_time_resolution(self):
        if(not self.isPhysicalTime()):
            return None;
        lARORder = {}
        lARORder[cTimeInfo.sRES_SECOND] = 60
        lARORder[cTimeInfo.sRES_MINUTE] = 60
        lARORder[cTimeInfo.sRES_HOUR] = 24
        lARORder[cTimeInfo.sRES_DAY] = 31
        lARORder[cTimeInfo.sRES_MONTH] = 12
        return lARORder.get(self.mResolution , None)
    
    def computeTimeDelta(self):
        #print(self.mSignalFrame.columns);
        # print(self.mSignalFrame[self.mTime].head());
        lEstim = self.mSplit.getEstimPart(self.mSignalFrame)
        lTimeBefore = lEstim[self.mTime].shift(1);
        # lTimeBefore.fillna(self.mTimeMin, inplace=True)
        N = lEstim.shape[0];
        if(N == 1):
            if(self.isPhysicalTime()):
                self.mTimeDelta = np.timedelta64(1,'D');
            else:
                self.mTimeDelta = 1
            return
        #print(self.mSignal, self.mTime, N);
        #print(lEstim[self.mTime].head());
        #print(lTimeBefore.head());
        lDiffs = lEstim[self.mTime][1:N] - lTimeBefore[1:N]
        
        if(self.mOptions.mTimeDeltaComputationMethod == "USER"):
            self.mTimeDelta = self.mOptions.mUserTimeDelta;
        if(self.mOptions.mTimeDeltaComputationMethod == "AVG"):
            self.mTimeDelta = np.mean(lDiffs);
            type1 = np.dtype(self.mSignalFrame[self.mTime])
            if(type1.kind == 'i' or type1.kind == 'u'):
                self.mTimeDelta = int(self.mTimeDelta)
        if(self.mOptions.mTimeDeltaComputationMethod == "MODE"):
            delta_counts = pd.DataFrame(lDiffs.value_counts());
            self.mTimeDelta = delta_counts[self.mTime].argmax();
        self.adaptTimeDeltaToTimeResolution();

    def estimate(self):
        #print(self.mSignalFrame.columns);
        #print(self.mSignalFrame[self.mTime].head());
        self.checkDateAndSignalTypes();
        
        self.mRowNumberColumn = "row_number"
        self.mNormalizedTimeColumn = self.mTime + "_Normalized";

        self.analyzeSeasonals();

        lEstim = self.mSplit.getEstimPart(self.mSignalFrame)
        self.mTimeMin = lEstim[self.mTime].min();
        self.mTimeMax = lEstim[self.mTime].max();
        if(self.isPhysicalTime()):
            self.mTimeMin = np.datetime64(self.mTimeMin.to_pydatetime());
            self.mTimeMax = np.datetime64(self.mTimeMax.to_pydatetime());
        self.mTimeMinMaxDiff = self.mTimeMax - self.mTimeMin;
        self.mEstimCount = lEstim.shape[0]
        # print(self.mTimeMin, self.mTimeMax , self.mTimeMinMaxDiff , (self.mTimeMax - self.mTimeMin)/self.mTimeMinMaxDiff)
        self.computeTimeDelta();
        self.mSignalFrame[self.mNormalizedTimeColumn] = self.compute_normalize_date_column(self.mSignalFrame[self.mTime])
        self.dump();

    def dump(self):
        time_info = self.info(); 
        

    def compute_normalize_date_column(self, idate_column):
        if(self.mEstimCount == 1):
            return 0.0;
        return idate_column.apply(self.normalizeTime)

    @tsutil.cMemoize
    def normalizeTime(self , iTime):
        if(self.mEstimCount == 1):
            return 0.0;
        output =  ( iTime- self.mTimeMin) / self.mTimeMinMaxDiff
        return output

    def nextTime(self, df, iSteps):
        #print(df.tail(1)[self.mTime]);
        lLastTime = df[self.mTime].values[-1]
        if(self.isPhysicalTime()):
            lLastTime = pd.Timestamp(lLastTime)
            # print("NEXT_TIME" , lLastTime, iSteps, self.mTimeDelta);
            lNextTime = lLastTime + iSteps * self.mTimeDelta;
            lNextTime = self.cast_to_time_dtype(lNextTime.to_datetime64())
        else:
            lNextTime = lLastTime + iSteps * self.mTimeDelta;
            lNextTime = self.cast_to_time_dtype(lNextTime)
            
            
        return lNextTime;
