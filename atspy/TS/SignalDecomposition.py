# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

import traceback

import dill
dill.settings['recurse'] = False
# import dill
# import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

# for timing
import time

import threading

from . import Time as tsti
from . import Exogenous as tsexog
from . import MissingData as tsmiss
from . import Signal_Transformation as tstransf
from . import Perf as tsperf
from . import SignalDecomposition_Trend as tstr
from . import SignalDecomposition_Cycle as tscy
from . import SignalDecomposition_AR as tsar
from . import Options as tsopts
from . import TimeSeriesModel as tsmodel
from . import TimeSeries_Cutting as tscut
from . import Utils as tsutil

import copy

class cPerf_Arg:
    def __init__(self , name):
        self.mName = name;
        self.mModel = None;
        self.mResult = None;

def compute_perf_func(arg):
    # print("RUNNING " , arg.mName)
    logger = tsutil.get_pyaf_logger();
    try:
        arg.mModel.updatePerfs();
        return arg;
    except Exception as e:
        # print("FAILURE_WITH_EXCEPTION : " + str(e)[:200]);
        logger.error("FAILURE_WITH_EXCEPTION : " + str(e)[:200]);
        logger.error("BENCHMARKING_FAILURE '" + arg.mName + "'");
        traceback.print_exc()
        raise;
    except:
        # print("BENCHMARK_FAILURE '" + arg.mName + "'");
        logger.error("BENCHMARK_FAILURE '" + arg.mName + "'");
        raise

class cSignalDecompositionOneTransform:
        
    def __init__(self):
        self.mSignalFrame = pd.DataFrame()
        self.mTime = None
        self.mSignal = None
        self.mTimeInfo = tsti.cTimeInfo();
        self.mForecastFrame = pd.DataFrame()
        self.mTransformation = tstransf.cSignalTransform_None();
        

    def serialize(self):
        from sklearn.externals import joblib
        joblib.dump(self, self.mTimeInfo.mTime + "_" + self.mSignal + "_TS.pkl")        

    def setParams(self , iInputDS, iTime, iSignal, iHorizon, iTransformation, iExogenousData = None):
        assert(iInputDS.shape[0] > 0)
        assert(iInputDS.shape[1] > 0)
        assert(iTime in iInputDS.columns)
        assert(iSignal in iInputDS.columns)

        # print("setParams , head", iInputDS.head());
        # print("setParams , tail", iInputDS.tail());
        # print("setParams , columns", iInputDS.columns);
        
        self.mTime = iTime
        self.mOriginalSignal = iSignal;
        
        self.mTransformation = iTransformation;
        self.mTransformation.mOriginalSignal = iSignal; 

        lMissingImputer = tsmiss.cMissingDataImputer()
        lMissingImputer.mOptions = self.mOptions
        lSignal = lMissingImputer.interpolate_signal_if_needed(iInputDS, iSignal)
        lTime = lMissingImputer.interpolate_time_if_needed(iInputDS, iTime)

        self.mTransformation.fit(lSignal);

        self.mSignal = iTransformation.get_name(iSignal)
        self.mHorizon = iHorizon;
        self.mSignalFrame = pd.DataFrame()
        self.mSignalFrame[self.mTime] = lTime;
        self.mSignalFrame[self.mOriginalSignal] = lSignal;
        self.mSignalFrame[self.mSignal] = self.mTransformation.apply(lSignal);
        self.mSignalFrame['row_number'] = np.arange(0, iInputDS.shape[0]);
        # self.mSignalFrame.dropna(inplace = True);
        assert(self.mSignalFrame.shape[0] > 0);

        # print("SIGNAL_INFO " , self.mSignalFrame.info());
        # print(self.mSignalFrame.head())


        self.mSplit = tscut.cCuttingInfo()
        self.mSplit.mTime = self.mTime;
        self.mSplit.mSignal = self.mSignal;
        self.mSplit.mOriginalSignal = self.mOriginalSignal;
        self.mSplit.mHorizon = self.mHorizon;
        self.mSplit.mSignalFrame = self.mSignalFrame;
        self.mSplit.mOptions = self.mOptions;
        
        
        self.mTimeInfo = tsti.cTimeInfo();
        self.mTimeInfo.mTime = self.mTime;
        self.mTimeInfo.mSignal = self.mSignal;
        self.mTimeInfo.mOriginalSignal = self.mOriginalSignal;
        self.mTimeInfo.mHorizon = self.mHorizon;
        self.mTimeInfo.mSignalFrame = self.mSignalFrame;
        self.mTimeInfo.mOptions = self.mOptions;
        self.mTimeInfo.mSplit = self.mSplit;

        self.mExogenousInfo = None;
        if(iExogenousData is not None):
            self.mExogenousInfo = tsexog.cExogenousInfo();
            self.mExogenousInfo.mExogenousData = iExogenousData;
            self.mExogenousInfo.mTimeInfo = self.mTimeInfo;
            self.mExogenousInfo.mOptions = self.mOptions;
        

    def computePerfsInParallel(self, args):
        lModels = {};
        # print([arg.mName for arg in args]);
        # print([arg.mModel.mOutName for arg in args]);
        pool = Pool(self.mOptions.mNbCores)
        # results = [compute_perf_func(arg) for arg in args];
        for res in pool.imap(compute_perf_func, args):
            # print("FINISHED_PERF_FOR_MODEL" , res.mName);
            lModels[res.mName] = res.mModel;
        
        # pool.close()
        # pool.join()
        return lModels;
            

    def updatePerfsForAllModels(self , iModels):
        self.mPerfsByModel = {}
        for model in iModels.keys():
            iModels[model].updatePerfs();
            
        for (name, model) in iModels.items():
            # print(name, model.__dict__);
            lComplexity = model.getComplexity();
            lFitPerf = model.mFitPerf;
            lForecastPerf = model.mForecastPerf;
            lTestPerf = model.mTestPerf;
            self.mPerfsByModel[model.mOutName] = [model, lComplexity, lFitPerf , lForecastPerf, lTestPerf];
        return iModels;

    def collectPerformanceIndices(self) :
        rows_list = [];

        logger = tsutil.get_pyaf_logger();
        
        for name in sorted(self.mPerfsByModel.keys()):
            lModel = self.mPerfsByModel[name][0];
            lComplexity = self.mPerfsByModel[name][1];
            lFitPerf = self.mPerfsByModel[name][2];
            lForecastPerf = self.mPerfsByModel[name][3];
            lTestPerf = self.mPerfsByModel[name][4];
            lModelCategory = lModel.get_model_category()
            row = [lModelCategory , lComplexity,
                   lFitPerf.mCount, lFitPerf.mL1,  lFitPerf.mL2, 
                   lFitPerf.mMAPE, lFitPerf.mMASE, 
                   lForecastPerf.mCount, lForecastPerf.mL1, lForecastPerf.mL2,
                   lForecastPerf.mMAPE, lForecastPerf.mMASE,
                   lTestPerf.mCount, lTestPerf.mL1, lTestPerf.mL2,
                   lTestPerf.mMAPE, lTestPerf.mMASE]
            rows_list.append(row);
            if(self.mOptions.mDebugPerformance):
                logger.debug("collectPerformanceIndices : " + str(row));
                
        self.mPerfDetails = pd.DataFrame(rows_list, columns=
                                         ('Model', 'Complexity',
                                          'FitCount', 'FitL1', 'FitL2', 'FitMAPE', 'FitMASE',
                                          'ForecastCount', 'ForecastL1', 'ForecastL2', 'ForecastMAPE',  'ForecastMASE', 
                                          'TestCount', 'TestL1', 'TestL2', 'TestMAPE', 'TestMASE')) 
        self.mPerfDetails.sort_values(by=['Forecast' + self.mOptions.mModelSelection_Criterion ,
                                          'Complexity', 'Model'] ,
                                      ascending=[True, True, True],
                                      inplace=True);
        self.mPerfDetails = self.mPerfDetails.reset_index(drop=True);
        # print(self.mPerfDetails.head());
        lBestName = self.mPerfDetails.iloc[0]['Model'];
        self.mBestModel = self.mPerfsByModel[lBestName][0];
        return self.mBestModel;
    

    def run_gc(self):
        import gc
        gc.collect()
    
    # @profile
    def train(self , iInputDS, iTime, iSignal,
              iHorizon, iTransformation):
        logger = tsutil.get_pyaf_logger();

        start_time = time.time()
        self.setParams(iInputDS, iTime, iSignal, iHorizon, iTransformation, self.mExogenousData);

        self.run_gc();

        # estimate time info
        # assert(self.mTimeInfo.mSignalFrame.shape[0] == iInputDS.shape[0])
        self.mSplit.estimate();
        self.mTimeInfo.estimate();


        exog_start_time = time.time()
        if(self.mExogenousInfo is not None):
            self.mExogenousInfo.fit();
            if(self.mOptions.mDebugProfile):
                logger.info("EXOGENOUS_ENCODING_TIME_IN_SECONDS " + str(self.mSignal) + " " + str(time.time() - exog_start_time))

        # estimate the trend

        lTrendEstimator = tstr.cTrendEstimator()
        lTrendEstimator.mSignalFrame = self.mSignalFrame
        lTrendEstimator.mTimeInfo = self.mTimeInfo
        lTrendEstimator.mSplit = self.mSplit
        lTrendEstimator.mOptions = self.mOptions;
        
        trend_start_time = time.time()
        lTrendEstimator.estimateTrend();
        #lTrendEstimator.plotTrend();
        if(self.mOptions.mDebugProfile):
            logger.info("TREND_TIME_IN_SECONDS "  + str(self.mSignal) + " " + str(time.time() - trend_start_time))


        # estimate cycles
        cycle_start_time = time.time()

        lCycleEstimator = tscy.cCycleEstimator();
        lCycleEstimator.mTrendFrame = lTrendEstimator.mTrendFrame;
        lCycleEstimator.mTrendList = lTrendEstimator.mTrendList;

        del lTrendEstimator;
        self.run_gc();

        lCycleEstimator.mTimeInfo = self.mTimeInfo
        lCycleEstimator.mSplit = self.mSplit
        lCycleEstimator.mOptions = self.mOptions;

        lCycleEstimator.estimateAllCycles();
        # if(self.mOptions.mDebugCycles):
            # lCycleEstimator.plotCycles();
        if(self.mOptions.mDebugProfile):
            logger.info("CYCLE_TIME_IN_SECONDS "  + str(self.mSignal) + " " + str( str(time.time() - cycle_start_time)))


        # autoregressive
        ar_start_time = time.time()
        lAREstimator = tsar.cAutoRegressiveEstimator();
        lAREstimator.mCycleFrame = lCycleEstimator.mCycleFrame;
        lAREstimator.mTrendList = lCycleEstimator.mTrendList;
        lAREstimator.mCycleList = lCycleEstimator.mCycleList;

        del lCycleEstimator;
        self.run_gc();

        lAREstimator.mTimeInfo = self.mTimeInfo
        lAREstimator.mSplit = self.mSplit
        lAREstimator.mExogenousInfo = self.mExogenousInfo;
        lAREstimator.mOptions = self.mOptions;
        lAREstimator.estimate();
        #lAREstimator.plotAR();
        if(self.mOptions.mDebugProfile):
            logger.info("AUTOREG_TIME_IN_SECONDS " + str(self.mSignal) + " " + str( str(time.time() - ar_start_time)))
        # forecast perfs

        perf_start_time = time.time()
        lModels = {};
        for trend in lAREstimator.mTrendList:
            for cycle in lAREstimator.mCycleList[trend]:
                cycle_residue = cycle.getCycleResidueName();
                for autoreg in lAREstimator.mARList[cycle_residue]:
                    lModel = tsmodel.cTimeSeriesModel(self.mTransformation, trend, cycle, autoreg);
                    lModels[lModel.mOutName] = lModel;

        del lAREstimator;
        self.updatePerfsForAllModels(lModels);
        
        if(self.mOptions.mDebugProfile):
            logger.info("PERF_TIME_IN_SECONDS " + str(self.mSignal) + " " + str(len(lModels)) + " " + str( str(time.time() - perf_start_time)))

        if(self.mOptions.mDebugProfile):
            logger.info("TRAINING_TIME_IN_SECONDS "  + str(self.mSignal) + " " + str(time.time() - start_time))
        self.run_gc();
        


class cTraining_Arg:
    def __init__(self , name):
        self.mName = name;
        self.mInputDS = None;
        self.mTime = None;
        self.mSignal = None;
        self.mHorizon = None;
        self.mTransformation = None;
        self.mSigDec = None;
        self.mResult = None;


def run_transform_thread(arg):
    # print("RUNNING_TRANSFORM", arg.mName);
    arg.mSigDec.train(arg.mInputDS, arg.mTime, arg.mSignal, arg.mHorizon, arg.mTransformation);
    return arg;

class cSignalDecompositionTrainer:
        
    def __init__(self):
        self.mSigDecByTransform = {};
        self.mOptions = tsopts.cSignalDecomposition_Options();
        self.mExogenousData = None;
        pass

    def train(self, iInputDS, iTime, iSignal, iHorizon):
        if(self.mOptions.mParallelMode):
            self.train_multiprocessed(iInputDS, iTime, iSignal, iHorizon);
        else:
            self.train_not_threaded(iInputDS, iTime, iSignal, iHorizon);
    

    def perform_model_selection(self):
        if(self.mOptions.mDebugPerformance):
            self.collectPerformanceIndices();
        else:
            self.collectPerformanceIndices_ModelSelection();

        self.cleanup_after_model_selection();

    
    def validateTransformation(self , transf , df, iTime, iSignal):
        lName = transf.get_name("");
        lIsApplicable = transf.is_applicable(df[iSignal]);
        if(lIsApplicable):
            # print("Adding Transformation " , lName);
            self.mTransformList = self.mTransformList + [transf];
    
    def defineTransformations(self , df, iTime, iSignal):
        self.mTransformList = [];
        if(self.mOptions.mActiveTransformations['None']):
            self.validateTransformation(tstransf.cSignalTransform_None() , df, iTime, iSignal);

        if(self.mOptions.mActiveTransformations['Difference']):
            self.validateTransformation(tstransf.cSignalTransform_Differencing() , df, iTime, iSignal);

        if(self.mOptions.mActiveTransformations['RelativeDifference']):
            self.validateTransformation(tstransf.cSignalTransform_RelativeDifferencing() , df, iTime, iSignal);
            
        if(self.mOptions.mActiveTransformations['Integration']):
            self.validateTransformation(tstransf.cSignalTransform_Accumulate() , df, iTime, iSignal);

        if(self.mOptions.mActiveTransformations['BoxCox']):
            for i in self.mOptions.mBoxCoxOrders:
                self.validateTransformation(tstransf.cSignalTransform_BoxCox(i) , df, iTime, iSignal);

        if(self.mOptions.mActiveTransformations['Quantization']):
            for q in self.mOptions.mQuantiles:
                self.validateTransformation(tstransf.cSignalTransform_Quantize(q) , df, iTime, iSignal);
        
        if(self.mOptions.mActiveTransformations['Logit']):
            self.validateTransformation(tstransf.cSignalTransform_Logit() , df, iTime, iSignal);
        
        if(self.mOptions.mActiveTransformations['Fisher']):
            self.validateTransformation(tstransf.cSignalTransform_Fisher() , df, iTime, iSignal);
        
        if(self.mOptions.mActiveTransformations['Anscombe']):
            self.validateTransformation(tstransf.cSignalTransform_Anscombe() , df, iTime, iSignal);
        

        for transform1 in self.mTransformList:
            transform1.mOptions = self.mOptions;
            transform1.mOriginalSignal = iSignal;
            transform1.test();

            
    def train_threaded(self , iInputDS, iTime, iSignal, iHorizon):
        threads = [] 
        self.defineTransformations(iInputDS, iTime, iSignal);
        for transform1 in self.mTransformList:
            t = threading.Thread(target=run_transform_thread,
                                 args = (iInputDS, iTime, iSignal, iHorizon, transform1, self.mOptions, self.mExogenousData))
            t.daemon = False
            threads += [t] 
            t.start()
 
        for t in threads: 
            t.join()
        
    def train_multiprocessed(self , iInputDS, iTime, iSignal, iHorizon):
        pool = Pool(self.mOptions.mNbCores)
        self.defineTransformations(iInputDS, iTime, iSignal);
        # print([transform1.mFormula for transform1 in self.mTransformList]);
        args = [];
        for transform1 in self.mTransformList:
            arg = cTraining_Arg(transform1.get_name(""));
            arg.mSigDec = cSignalDecompositionOneTransform();
            arg.mSigDec.mOptions = self.mOptions;
            arg.mSigDec.mExogenousData = self.mExogenousData;
            arg.mInputDS = iInputDS;
            arg.mTime = iTime;
            arg.mSignal = iSignal;
            arg.mHorizon = iHorizon;
            arg.mTransformation = transform1;
            arg.mOptions = self.mOptions;
            arg.mExogenousData = self.mExogenousData;
            arg.mResult = None;
            
            args.append(arg);

        for res in pool.imap(run_transform_thread, args):
            # print("FINISHED_TRAINING" , res.mName);
            self.mSigDecByTransform[res.mTransformation.get_name("")] = res.mSigDec;
        # pool.close()
        # pool.join()
        
	
        
    def train_not_threaded(self , iInputDS, iTime, iSignal, iHorizon):
        self.defineTransformations(iInputDS, iTime, iSignal);
        for transform1 in self.mTransformList:
            sigdec = cSignalDecompositionOneTransform();
            sigdec.mOptions = self.mOptions;
            sigdec.mExogenousData = self.mExogenousData;
            sigdec.train(iInputDS, iTime, iSignal, iHorizon, transform1);
            self.mSigDecByTransform[transform1.get_name("")] = sigdec


    def collectPerformanceIndices_ModelSelection(self) :
        modelsel_start_time = time.time()
        logger = tsutil.get_pyaf_logger();

        rows_list = []
        self.mPerfsByModel = {}
        for transform1 in self.mTransformList:
            sigdec = self.mSigDecByTransform[transform1.get_name("")]
            for (model , value) in sorted(sigdec.mPerfsByModel.items()):
                self.mPerfsByModel[model] = value
                lTranformName = sigdec.mSignal;
                lModelFormula = model
                lModelCategory = value[0].get_model_category()
                lSplit = value[0].mTimeInfo.mOptions.mCustomSplit
                #  value format : self.mPerfsByModel[lModel.mOutName] = [lModel, lComplexity, lFitPerf , lForecastPerf, lTestPerf];
                lComplexity = value[1];
                lFitPerf = value[2];
                lForecastPerf = value[3];
                lTestPerf = value[4];
                row = [lSplit, lTranformName, lModelFormula , lModelCategory, lComplexity,
                       lFitPerf.getCriterionValue(self.mOptions.mModelSelection_Criterion),
                       lForecastPerf.getCriterionValue(self.mOptions.mModelSelection_Criterion),
                       lTestPerf.getCriterionValue(self.mOptions.mModelSelection_Criterion)]
                rows_list.append(row);
                if(self.mOptions.mDebugPerformance):
                    logger.info("collectPerformanceIndices : " + self.mOptions.mModelSelection_Criterion + " " +  str(row[0]) + " " + str(row[2]) + " " + str(row[4]) + " " +str(row[7]));

        self.mTrPerfDetails =  pd.DataFrame(rows_list, columns=
                                            ('Split', 'Transformation', 'Model', 'Category', 'Complexity',
                                             'Fit' + self.mOptions.mModelSelection_Criterion,
                                             'Forecast' + self.mOptions.mModelSelection_Criterion,
                                             'Test' + self.mOptions.mModelSelection_Criterion)) 
        # print(self.mTrPerfDetails.head(self.mTrPerfDetails.shape[0]));
        lIndicator = 'Forecast' + self.mOptions.mModelSelection_Criterion;
        lBestPerf = self.mTrPerfDetails[ lIndicator ].min();
        # allow a loss of one point (0.01 of MAPE) if complexity is reduced.
        if(not np.isnan(lBestPerf)):
            self.mTrPerfDetails.sort_values(by=[lIndicator, 'Complexity', 'Model'] ,
                                            ascending=[True, True, True],
                                            inplace=True);
            self.mTrPerfDetails = self.mTrPerfDetails.reset_index(drop=True);
                
            lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails[lIndicator] <= (lBestPerf + 0.01)].reset_index(drop=True);
        else:
            lInterestingModels = self.mTrPerfDetails;
        lInterestingModels.sort_values(by=['Complexity'] , ascending=True, inplace=True)
        # print(self.mTransformList);
        # print(lInterestingModels.head());
        # print(self.mPerfsByModel);
        lBestName = lInterestingModels['Model'].iloc[0];
        self.mBestModel = self.mPerfsByModel[lBestName][0];
        # print(lBestName, self.mBestModel)
        if(self.mOptions.mDebugProfile):
            logger.info("MODEL_SELECTION_TIME_IN_SECONDS "  + str(self.mBestModel.mSignal) + " " + str(time.time() - modelsel_start_time))

    def collectPerformanceIndices(self) :
        modelsel_start_time = time.time()
        logger = tsutil.get_pyaf_logger();

        rows_list = []
        self.mPerfsByModel = {}
        for transform1 in self.mTransformList:
            sigdec = self.mSigDecByTransform[transform1.get_name("")]
            for (model , value) in sorted(sigdec.mPerfsByModel.items()):
                self.mPerfsByModel[model] = value;
                lTranformName = sigdec.mSignal;
                lModelFormula = model
                lModelCategory = value[0].get_model_category()
                lSplit = value[0].mTimeInfo.mOptions.mCustomSplit
                #  value format : self.mPerfsByModel[lModel.mOutName] = [lModel, lComplexity, lFitPerf , lForecastPerf, lTestPerf];
                lComplexity = value[1];
                lFitPerf = value[2];
                lForecastPerf = value[3];
                lTestPerf = value[4];
                row = [lSplit, lTranformName, lModelFormula , lModelCategory, lComplexity,
                       lFitPerf.mCount, lFitPerf.mL1, lFitPerf.mL2, lFitPerf.mMAPE,  lFitPerf.mMASE, 
                       lForecastPerf.mCount, lForecastPerf.mL1, lForecastPerf.mL2, lForecastPerf.mMAPE, lForecastPerf.mMASE,
                       lTestPerf.mCount, lTestPerf.mL1, lTestPerf.mL2, lTestPerf.mMAPE, lTestPerf.mMASE]
                rows_list.append(row);
                if(self.mOptions.mDebugPerformance):
                    lIndicatorValue = lForecastPerf.getCriterionValue(self.mOptions.mModelSelection_Criterion)
                    logger.info("collectPerformanceIndices : " + self.mOptions.mModelSelection_Criterion + " " + str(row[0])+ " " + str(row[1]) + " " + str(row[3]) + " " + str(row[4]) + " " + str(lIndicatorValue));

        self.mTrPerfDetails =  pd.DataFrame(rows_list, columns=
                                            ('Split', 'Transformation', 'Model', 'Category', 'Complexity',
                                             'FitCount', 'FitL1', 'FitL2', 'FitMAPE', 'FitMASE',
                                             'ForecastCount', 'ForecastL1', 'ForecastL2', 'ForecastMAPE', 'ForecastMASE',
                                             'TestCount', 'TestL1', 'TestL2', 'TestMAPE', 'TestMASE')) 
        # print(self.mTrPerfDetails.head(self.mTrPerfDetails.shape[0]));
        lIndicator = 'Forecast' + self.mOptions.mModelSelection_Criterion;
        lBestPerf = self.mTrPerfDetails[ lIndicator ].min();
        # allow a loss of one point (0.01 of MAPE) if complexity is reduced.
        if(not np.isnan(lBestPerf)):
            self.mTrPerfDetails.sort_values(by=[lIndicator, 'Complexity', 'Model'] ,
                                            ascending=[True, True, True],
                                            inplace=True);
            self.mTrPerfDetails = self.mTrPerfDetails.reset_index(drop=True);
                
            lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails[lIndicator] <= (lBestPerf + 0.01)].reset_index(drop=True);
        else:
            lInterestingModels = self.mTrPerfDetails;
        lInterestingModels.sort_values(by=['Complexity'] , ascending=True, inplace=True)
        # print(self.mTransformList);
        print(lInterestingModels.head());
        lBestName = lInterestingModels['Model'].iloc[0];
        self.mBestModel = self.mPerfsByModel[lBestName][0];
        if(self.mOptions.mDebugProfile):
            logger.info("MODEL_SELECTION_TIME_IN_SECONDS "  + str(self.mBestModel.mSignal) + " " + str(time.time() - modelsel_start_time))

            
    def cleanup_after_model_selection(self):
        lBestTransformationName = self.mBestModel.mTransformation.get_name("")
        lSigDecByTransform = {}
        for (name, sigdec) in self.mSigDecByTransform.items():
            if(name == lBestTransformationName):
                for modelname in sigdec.mPerfsByModel.keys():
                    # store only model names here.
                    sigdec.mPerfsByModel[modelname][0] = modelname
                    lSigDecByTransform[name]  = sigdec                
        # delete failing transformations
        del self.mSigDecByTransform
        self.mSigDecByTransform = lSigDecByTransform
        

class cSignalDecompositionTrainer_CrossValidation:
    def __init__(self):
        self.mSigDecByTransform = {};
        self.mOptions = tsopts.cSignalDecomposition_Options();
        self.mExogenousData = None;
        pass

    def define_splits(self):
        lFolds = self.mOptions.mCrossValidationOptions.mNbFolds
        lRatio = 1.0 / lFolds
        lSplits = [(k * lRatio , lRatio , 0.0) for k in range(lFolds // 2, lFolds)]
        return lSplits
    
    def train(self, iInputDS, iTime, iSignal, iHorizon):
        cross_val_start_time = time.time()
        logger = tsutil.get_pyaf_logger();
        self.mSplits = self.define_splits()
        self.mTrainers = {}
        for lSplit in self.mSplits:
            split_start_time = time.time()
            logger.info("CROSS_VALIDATION_TRAINING_SIGNAL_SPLIT '" + iSignal + "' " + str(lSplit));
            lTrainer = cSignalDecompositionTrainer()        
            lTrainer.mOptions = copy.copy(self.mOptions);
            lTrainer.mOptions.mCustomSplit = lSplit
            lTrainer.mExogenousData = self.mExogenousData;
            lTrainer.train(iInputDS, iTime, iSignal, iHorizon)
            lTrainer.collectPerformanceIndices_ModelSelection()
            self.mTrainers[lSplit] = lTrainer
            logger.info("CROSS_VALIDATION_TRAINING_SPLIT_TIME_IN_SECONDS '" + iSignal + "' " + str(lSplit) + " " + str(time.time() - split_start_time))
        self.perform_model_selection()
        logger.info("CROSS_VALIDATION_TRAINING_TIME_IN_SECONDS "  + str(self.mBestModel.mSignal) + " " + str(time.time() - cross_val_start_time))

    def perform_model_selection(self):
        logger = tsutil.get_pyaf_logger();
        modelsel_start_time = time.time()
        self.mTrPerfDetails = pd.DataFrame()
        for (lSplit , lTrainer) in self.mTrainers.items():
            self.mTrPerfDetails = self.mTrPerfDetails.append(lTrainer.mTrPerfDetails)
        # self.mTrPerfDetails.to_csv("perf_time_series_cross_val.csv")
        lIndicator = 'Forecast' + self.mOptions.mModelSelection_Criterion;
        lColumns = ['Category', 'Complexity', lIndicator]
        lPerfByCategory = self.mTrPerfDetails[lColumns].groupby(by=['Category'] , sort=False)[lIndicator].mean()
        lPerfByCategory_df = pd.DataFrame(lPerfByCategory).reset_index()
        lPerfByCategory_df.columns = ['Category' , lIndicator]
        # lPerfByCategory_df.to_csv("perf_time_series_cross_val_by_category.csv")
        lBestPerf = lPerfByCategory_df[ lIndicator ].min();
        lPerfByCategory_df.sort_values(by=[lIndicator, 'Category'] ,
                                ascending=[True, True],
                                inplace=True);
        lPerfByCategory_df = lPerfByCategory_df.reset_index(drop=True);
                
        lInterestingCategories_df = lPerfByCategory_df[lPerfByCategory_df[lIndicator] <= (lBestPerf + 0.01)].reset_index(drop=True);
        # print(lPerfByCategory_df.head());
        # print(lInterestingCategories_df.head());
        # print(self.mPerfsByModel);
        lInterestingCategories = list(lInterestingCategories_df['Category'].unique())
        self.mTrPerfDetails['IC'] = self.mTrPerfDetails['Category'].apply(lambda x :1 if x in lInterestingCategories else 0) 
        lInterestingModels = self.mTrPerfDetails[self.mTrPerfDetails['IC'] == 1].copy()
        lInterestingModels.sort_values(by=['Complexity'] , ascending=True, inplace=True)
        # print(self.mTransformList);
        # print(lInterestingModels.head());
        lBestName = lInterestingModels['Model'].iloc[0];
        lBestSplit = lInterestingModels['Split'].iloc[0];
        # print(self.mTrainers.keys())
        lBestTrainer = self.mTrainers[lBestSplit]
        self.mBestModel = lBestTrainer.mPerfsByModel[lBestName][0];
        # print(lBestName, self.mBestModel)
        if(self.mOptions.mDebugProfile):
            logger.info("MODEL_SELECTION_TIME_IN_SECONDS "  + str(self.mBestModel.mSignal) + " " + str(time.time() - modelsel_start_time))
        pass
    

class cSignalDecomposition:
        
    def __init__(self):
        self.mSigDecByTransform = {};
        self.mOptions = tsopts.cSignalDecomposition_Options();
        self.mExogenousData = None;
        pass

    def checkData(self, iInputDS, iTime, iSignal, iHorizon, iExogenousData):        
        if(iHorizon != int(iHorizon)):
            raise tsutil.PyAF_Error("PYAF_ERROR_NON_INTEGER_HORIZON " + str(iHorizon));
        if(iHorizon < 1):
            raise tsutil.PyAF_Error("PYAF_ERROR_NEGATIVE_OR_NULL_HORIZON " + str(iHorizon));
        if(iTime not in iInputDS.columns):
            raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_NOT_FOUND " + str(iTime));
        if(iSignal not in iInputDS.columns):
            raise tsutil.PyAF_Error("PYAF_ERROR_SIGNAL_COLUMN_NOT_FOUND " + str(iSignal));
        type1 = np.dtype(iInputDS[iTime])
        # print(type1)
        if(type1.kind != 'M' and type1.kind != 'i' and type1.kind != 'u' and type1.kind != 'f'):
            raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_TYPE_NOT_ALLOWED '" + str(iTime) + "' '" + str(type1) + "'");
        type2 = np.dtype(iInputDS[iSignal])
        # print(type2)
        if(type2.kind != 'i' and type2.kind != 'u' and type2.kind != 'f'):
            raise tsutil.PyAF_Error("PYAF_ERROR_SIGNAL_COLUMN_TYPE_NOT_ALLOWED '" + str(iSignal) + "' '" + str(type2) + "'");
        # time in exogenous data should be the strictly same type as time in training dataset (join needed)
        if(iExogenousData is not None):
            lExogenousDataFrame = iExogenousData[0];
            lExogenousVariables = iExogenousData[1];
            if(iTime not in lExogenousDataFrame.columns):
                raise tsutil.PyAF_Error("PYAF_ERROR_TIME_COLUMN_NOT_FOUND_IN_EXOGENOUS " + str(iTime));
            for exog in lExogenousVariables:
                if(exog not in lExogenousDataFrame.columns):
                    raise tsutil.PyAF_Error("PYAF_ERROR_EXOGENOUS_VARIABLE_NOT_FOUND " + str(exog));
                
            type3 = np.dtype(lExogenousDataFrame[iTime])
            if(type1 != type3):
                raise tsutil.PyAF_Error("PYAF_ERROR_INCOMPATIBLE_TIME_COLUMN_TYPE_IN_EXOGENOUS '" + str(iTime) + "' '" + str(type1)  + "' '" + str(type3) + "'");
                
    # @profile
    def train(self , iInputDS, iTime, iSignal, iHorizon, iExogenousData = None):
        logger = tsutil.get_pyaf_logger();
        logger.info("START_TRAINING '" + str(iSignal) + "'")
        start_time = time.time()

        self.checkData(iInputDS, iTime, iSignal, iHorizon, iExogenousData);

        self.mTrainingDataset = iInputDS; 
        self.mExogenousData = iExogenousData;

        lTrainer = cSignalDecompositionTrainer()
        if(self.mOptions.mCrossValidationOptions.mMethod is not None):
            lTrainer = cSignalDecompositionTrainer_CrossValidation()        
        lTrainer.mOptions = self.mOptions;
        lTrainer.mExogenousData = iExogenousData;
        lTrainer.train(iInputDS, iTime, iSignal, iHorizon)
        lTrainer.perform_model_selection()
        self.mBestModel = lTrainer.mBestModel
        self.mTrPerfDetails = lTrainer.mTrPerfDetails
        
        # Prediction Intervals
        pred_interval_start_time = time.time()
        self.mBestModel.updatePerfs(compute_all_indicators = True);
        self.mBestModel.computePredictionIntervals();
        if(self.mOptions.mDebugProfile):
            logger.info("PREDICTION_INTERVAL_TIME_IN_SECONDS "  + str(iSignal) + " " + str(time.time() - pred_interval_start_time))

        end_time = time.time()
        self.mTrainingTime = end_time - start_time;
        logger.info("END_TRAINING_TIME_IN_SECONDS '" + str(iSignal) + "' " + str(self.mTrainingTime))
        pass

    def forecast(self , iInputDS, iHorizon):
        logger = tsutil.get_pyaf_logger();
        logger.info("START_FORECASTING")
        start_time = time.time()
        lMissingImputer = tsmiss.cMissingDataImputer()
        lMissingImputer.mOptions = self.mOptions
        lInputDS = iInputDS.copy()
        lInputDS[self.mBestModel.mOriginalSignal] = lMissingImputer.interpolate_signal_if_needed(iInputDS, self.mBestModel.mOriginalSignal)
        lInputDS[self.mBestModel.mTime] = lMissingImputer.interpolate_time_if_needed(iInputDS, self.mBestModel.mTime)
        lForecastFrame = self.mBestModel.forecast(lInputDS, iHorizon);
        lForecastTime = time.time() - start_time;
        logger.info("END_FORECAST_TIME_IN_SECONDS " + str(lForecastTime))
        return lForecastFrame;


    def getModelFormula(self):
        lFormula = self.mBestModel.getFormula();
        return lFormula;


    def getModelInfo(self):
        return self.mBestModel.getInfo();

    def to_json(self):
        dict1 = self.mBestModel.to_json();
        import json
        return json.dumps(dict1, indent=4, sort_keys=True);
        
    def standardPlots(self, name = None, format = 'png'):
        logger = tsutil.get_pyaf_logger();
        logger.info("START_PLOTTING")
        start_time = time.time()
        self.mBestModel.standardPlots(name, format);
        lPlotTime = time.time() - start_time;
        logger.info("END_PLOTTING_TIME_IN_SECONDS " + str(lPlotTime))
        
    def getPlotsAsDict(self):
        lDict = self.mBestModel.getPlotsAsDict();
        return lDict;
