# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

from . import PredictionIntervals as predint

from . import Plots as tsplot
from . import Perf as tsperf
from . import Utils as tsutil

class cTimeSeriesModel:
    
    def __init__(self, transf, trend, cycle, autoreg):
        self.mTransformation = transf;
        self.mTrend = trend;
        self.mCycle = cycle;
        self.mAR = autoreg;
        self.mFitPerformances = {}
        self.mForecastPerformances = {}
        self.mTestPerformances = {}
        self.mOutName = self.mAR.mOutName;
        self.mOriginalSignal = self.mTransformation.mOriginalSignal;
        self.mTimeInfo = self.mTrend.mTimeInfo;
        self.mTime = self.mTimeInfo.mTime;
        self.mSignal = self.mTimeInfo.mSignal;
        self.mTrainingVersionInfo = self.getVersions();

    def signal_info(self):
        lSignal = self.mTrend.mSignalFrame[self.mOriginalSignal];
        lStr1 = "SignalVariable='" + self.mOriginalSignal +"'";
        lStr1 += " Min=" + str(np.min(lSignal)) + " Max="  + str(np.max(lSignal)) + " ";
        lStr1 += " Mean=" + str(np.mean(lSignal)) + " StdDev="  + str(np.std(lSignal));
        lSignal = self.mTrend.mSignalFrame[self.mSignal];
        lStr2 = "TransformedSignalVariable='" + self.mSignal +"'";
        lStr2 += " Min=" + str(np.min(lSignal)) + " Max="  + str(np.max(lSignal)) + " ";
        lStr2 += " Mean=" + str(np.mean(lSignal)) + " StdDev="  + str(np.std(lSignal));
        return (lStr1 , lStr2);

    def get_model_category(self):
        lModelCategory = (self.mTransformation.__class__.__name__,
                          self.mTrend.__class__.__name__,
                          self.mCycle.__class__.__name__,
                          self.mAR.__class__.__name__)
        lModelCategory = self.mTransformation.mFormula + "_" + self.mTrend.mFormula + "_" + self.mCycle.mFormula + "_" + self.mAR.mFormula
        return str(lModelCategory)
        
    def getComplexity(self):
        lComplexity = 32 * self.mTransformation.mComplexity +  16 * self.mTrend.mComplexity + 4 * self.mCycle.mComplexity + 1 * self.mAR.mComplexity;
        return lComplexity;     

    def updatePerfs(self, compute_all_indicators = False):
        self.mModelFrame = pd.DataFrame();
        lSignal = self.mTrend.mSignalFrame[self.mSignal]
        N = lSignal.shape[0];
        self.mTrend.mTimeInfo.addVars(self.mModelFrame);
        df = self.forecastOneStepAhead(self.mModelFrame , perf_mode = True);
        self.mModelFrame = df.head(N);
        # print(self.mModelFrame.columns);
        lPrefix = self.mSignal + "_";
        lForecastColumnName = str(self.mOriginalSignal) + "_Forecast";
        lFitPerf = tsperf.cPerf();
        lForecastPerf = tsperf.cPerf();
        lTestPerf = tsperf.cPerf();
        # self.mModelFrame.to_csv(self.mOutName + "_model_perf.csv");
        (lFrameFit, lFrameForecast, lFrameTest) = self.mTrend.mSplit.cutFrame(self.mModelFrame);


        if(compute_all_indicators):
            lFitPerf.compute(lFrameFit[self.mOriginalSignal] , lFrameFit[lForecastColumnName] ,
                             self.mOutName + '_Fit')
            lForecastPerf.compute(lFrameForecast[self.mOriginalSignal] ,
                                  lFrameForecast[lForecastColumnName],
                                  self.mOutName + '_Forecast')
            if(lFrameTest.shape[0] > 0):
                lTestPerf.compute(lFrameTest[self.mOriginalSignal] , lFrameTest[lForecastColumnName],
                                  self.mOutName + '_Test')            
            pass
        else:
            lFitPerf.computeCriterion(lFrameFit[self.mOriginalSignal] , lFrameFit[lForecastColumnName] ,
                                      self.mTimeInfo.mOptions.mModelSelection_Criterion,
                                      self.mOutName + '_Fit')
            lForecastPerf.computeCriterion(lFrameForecast[self.mOriginalSignal] , lFrameForecast[lForecastColumnName],
                                           self.mTimeInfo.mOptions.mModelSelection_Criterion,
                                           self.mOutName + '_Forecast')
            if(lFrameTest.shape[0] > 0):
                lTestPerf.computeCriterion(lFrameTest[self.mOriginalSignal] , lFrameTest[lForecastColumnName],
                                           self.mTimeInfo.mOptions.mModelSelection_Criterion,
                                           self.mOutName + '_Test')
            
        self.mFitPerf = lFitPerf
        self.mForecastPerf = lForecastPerf;
        self.mTestPerf = lTestPerf;
        # print("PERF_COMPUTATION" , self.mOutName, self.mFitPerf.mMAPE);
        
    def computePredictionIntervals(self):
        # prediction intervals
        if(self.mTimeInfo.mOptions.mAddPredictionIntervals):
            self.mPredictionIntervalsEstimator = predint.cPredictionIntervalsEstimator();
            self.mPredictionIntervalsEstimator.mModel = self;        
            self.mPredictionIntervalsEstimator.computePerformances();
        pass

    def getFormula(self):
        lFormula = self.mTrend.mFormula + " + ";
        lFormula += self.mCycle.mFormula + " + ";
        lFormula += self.mAR.mFormula;
        return lFormula;


    def getInfo(self):
        logger = tsutil.get_pyaf_logger();
        logger.info("TIME_DETAIL " + self.mTrend.mTimeInfo.info());
        sig_info = self.signal_info()
        logger.info("SIGNAL_DETAIL_ORIG " + sig_info[0]);
        logger.info("SIGNAL_DETAIL_TRANSFORMED " + sig_info[1]);
        logger.info("BEST_TRANSOFORMATION_TYPE '" + self.mTransformation.get_name("") + "'");
        logger.info("BEST_DECOMPOSITION  '" + self.mOutName + "' [" + self.getFormula() + "]");
        logger.info("TREND_DETAIL '" + self.mTrend.mOutName + "' [" + self.mTrend.mFormula + "]");
        logger.info("CYCLE_DETAIL '"+ self.mCycle.mOutName + "' [" + self.mCycle.mFormula + "]");
        logger.info("AUTOREG_DETAIL '" + self.mAR.mOutName + "' [" + self.mAR.mFormula + "]");
        logger.info("MODEL_MAPE MAPE_Fit=" + str(self.mFitPerf.mMAPE) + " MAPE_Forecast=" + str(self.mForecastPerf.mMAPE)  + " MAPE_Test=" + str(self.mTestPerf.mMAPE) );
        logger.info("MODEL_SMAPE SMAPE_Fit=" + str(self.mFitPerf.mSMAPE) + " SMAPE_Forecast=" + str(self.mForecastPerf.mSMAPE)  + " SMAPE_Test=" + str(self.mTestPerf.mSMAPE) );
        logger.info("MODEL_MASE MASE_Fit=" + str(self.mFitPerf.mMASE) + " MASE_Forecast=" + str(self.mForecastPerf.mMASE)  + " MASE_Test=" + str(self.mTestPerf.mMASE) );
        logger.info("MODEL_L1 L1_Fit=" + str(self.mFitPerf.mL1) + " L1_Forecast=" + str(self.mForecastPerf.mL1)  + " L1_Test=" + str(self.mTestPerf.mL1) );
        logger.info("MODEL_L2 L2_Fit=" + str(self.mFitPerf.mL2) + " L2_Forecast=" + str(self.mForecastPerf.mL2)  + " L2_Test=" + str(self.mTestPerf.mL2) );
        logger.info("MODEL_COMPLEXITY " + str(self.getComplexity()) );
        logger.info("AR_MODEL_DETAIL_START");
        self.mAR.dumpCoefficients();
        logger.info("AR_MODEL_DETAIL_END");



    def forecastOneStepAhead(self , df , horizon_index = 1, perf_mode = False):
        assert(self.mTime in df.columns)
        assert(self.mOriginalSignal in df.columns)
        lPrefix = self.mSignal + "_";
        df1 = df;
        # df1.to_csv("before.csv");
        # add new line with new time value, row_number and nromalized time
        # add signal tranformed column
        df1 = self.mTransformation.transformDataset(df1, self.mOriginalSignal);
        # df1.to_csv("after_transformation.csv");
        #print("Transformation update : " , df1.columns);

        df1 = self.mTimeInfo.transformDataset(df1);
        # df1.to_csv("after_time.csv");
        # print("TimeInfo update : " , df1.columns);
        # compute the trend based on the transformed column and compute trend residue
        df1 = self.mTrend.transformDataset(df1);
        #print("Trend update : " , df1.columns);
        # df1.to_csv("after_trend.csv");
        # compute the cycle and its residue based on the trend residue
        df1 = self.mCycle.transformDataset(df1);
        # df1.to_csv("after_cycle.csv");
        #print("Cycle update : " , df1.columns);
        # compute the AR componnet and its residue based on the cycle residue
        df1 = self.mAR.transformDataset(df1, horizon_index);
        # df1.to_csv("after_ar.csv");
        #print("AR update : " , df1.columns);
        # compute the forecast and its residue (forecast = trend  + cycle + AR)
        df2 = df1;
        lTrendColumn = df2[self.mTrend.mOutName]
        lCycleColumn = df2[self.mCycle.mOutName]
        lARColumn = df2[self.mAR.mOutName]
        lSignal = df2[self.mSignal]
        if(not perf_mode):
            df2[lPrefix + 'Trend'] =  lTrendColumn;
            df2[lPrefix + 'Trend_residue'] =  lSignal - lTrendColumn;
            df2[lPrefix + 'Cycle'] =  lCycleColumn;
            df2[lPrefix + 'Cycle_residue'] = df2[lPrefix + 'Trend_residue'] - lCycleColumn;
            df2[lPrefix + 'AR'] =  lARColumn ;
            df2[lPrefix + 'AR_residue'] = df2[lPrefix + 'Cycle_residue'] - lARColumn;

        lPrefix2 = str(self.mOriginalSignal) + "_";
        # print("TimeSeriesModel_forecast_invert");
        df2[lPrefix + 'TransformedForecast'] =  lTrendColumn + lCycleColumn + lARColumn ;
        df2[lPrefix2 + 'Forecast'] = self.mTransformation.invert(df2[lPrefix + 'TransformedForecast']);

        if(not perf_mode):
            df2[lPrefix + 'TransformedResidue'] =  lSignal - df2[lPrefix + 'TransformedForecast']
            lOriginalSignal = df2[self.mOriginalSignal]
            df2[lPrefix2 + 'Residue'] =  lOriginalSignal - df2[lPrefix2 + 'Forecast']
        df2.reset_index(drop=True, inplace=True);
        return df2;


    def forecast(self , df , iHorizon):
        N0 = df.shape[0];
        df1 = self.forecastOneStepAhead(df, 1)
        lForecastColumnName = str(self.mOriginalSignal) + "_Forecast";
        for h in range(0 , iHorizon - 1):
            # print(df1.info());
            N = df1.shape[0];
            # replace the signal with the forecast in the last line  of the dataset
            lPos = df1.index[N - 1];
            lSignal = df1.loc[lPos , lForecastColumnName];
            df1.loc[lPos , self.mOriginalSignal] = lSignal;
            df1 = df1[[self.mTime , self.mOriginalSignal]];
            df1 = self.forecastOneStepAhead(df1 , h+2)

        assert((N0 + iHorizon) == df1.shape[0])
        N1 = df1.shape[0];
        lPrefix = self.mSignal + "_";
        for h in range(0 , iHorizon):
            df1.loc[N1 - 1 - h, self.mSignal] = np.nan;
            df1.loc[N1 - 1 - h, self.mOriginalSignal] = np.nan;
            df1.loc[N1 - 1 - h, self.mTrend.mOutName + '_residue'] =  np.nan;
            df1.loc[N1 - 1 - h, self.mCycle.mOutName + '_residue'] =  np.nan;
            df1.loc[N1 - 1 - h, self.mAR.mOutName + '_residue'] =  np.nan;
            df1.loc[N1 - 1 - h, lPrefix + 'Trend_residue'] =  np.nan;
            df1.loc[N1 - 1 - h, lPrefix + 'Cycle_residue'] = np.nan;
            df1.loc[N1 - 1 - h, lPrefix + 'AR_residue'] = np.nan;
            df1.loc[N1 - 1 - h, str(self.mOriginalSignal) + '_Residue'] = np.nan;
            df1.loc[N1 - 1 - h, lPrefix + 'TransformedResidue'] = np.nan;
            pass
        # print(df.head())
        # print(df1.head())
        if(self.mTimeInfo.mOptions.mAddPredictionIntervals):
            df1 = self.addPredictionIntervals(df, df1, iHorizon);
        if(self.mTimeInfo.mOptions.mForecastRectifier is not None):
            df1 = self.applyForecastRectifier(df1)
        return df1

    def applyForecastRectifier(self, df):
        df1 = df;
        if(self.mTimeInfo.mOptions.mForecastRectifier == "relu"):
            lForecastColumnName = str(self.mOriginalSignal) + "_Forecast";
            df1[lForecastColumnName] = df1[lForecastColumnName].apply(lambda x : max(x, 0))
        return df1

    def addPredictionIntervals(self, iInputDS, iForecastFrame, iHorizon):
        lSignalColumn = self.mOriginalSignal;

        N = iInputDS.shape[0];
        lForecastColumn = str(lSignalColumn) + "_Forecast";
        lLowerBoundName = lForecastColumn + '_Lower_Bound'
        lUpperBoundName = lForecastColumn + '_Upper_Bound';
        iForecastFrame[lLowerBoundName] = np.nan; 
        iForecastFrame[lUpperBoundName] = np.nan; 

        lConfidence = 1.96 ; # 0.95
        # the prediction intervals are only computed for the training horizon
        lHorizon = min(iHorizon , self.mTimeInfo.mHorizon);
        for h in range(0 , lHorizon):
            lHorizonName = lForecastColumn + "_" + str(h + 1);
            lWidth = lConfidence * self.mPredictionIntervalsEstimator.mForecastPerformances[lHorizonName].mL2;
            iForecastFrame.loc[N + h , lLowerBoundName] = iForecastFrame.loc[N + h , lForecastColumn] - lWidth;
            iForecastFrame.loc[N + h , lUpperBoundName] = iForecastFrame.loc[N + h , lForecastColumn] + lWidth;
            
        return iForecastFrame;


    def plotForecasts(self, df):
        lPrefix = self.mSignal + "_";
        lTime = self.mTimeInfo.mNormalizedTimeColumn;            
        tsplot.decomp_plot(df,
                           lTime, lPrefix + 'Signal',
                           lPrefix + 'Forecast' , lPrefix + 'Residue');

    def to_json(self):
        dict1 = {};
        d1 = { "Time" : self.mTimeInfo.to_json(),
               "Signal" : self.mOriginalSignal,
               "Training_Signal_Length" : self.mModelFrame.shape[0]};
        dict1["Dataset"] = d1;
        lTransformation = self.mTransformation.mFormula;
        d2 = { "Best_Decomposition" : self.mOutName,
               "Signal_Transoformation" : lTransformation,
               "Trend" : self.mTrend.mFormula,
               "Cycle" : self.mCycle.mFormula,
               "AR_Model" : self.mAR.mFormula,
               };
        dict1["Model"] = d2;
        d3 = {"MAPE" : str(self.mForecastPerf.mMAPE),
              "MASE" : str(self.mForecastPerf.mMASE),
              "MAE" : str(self.mForecastPerf.mL1),
              "RMSE" : str(self.mForecastPerf.mL2),
              "COMPLEXITY" : str(self.getComplexity())};
        dict1["Model_Performance"] = d3;
        return dict1;


    def getForecastDatasetForPlots(self):
        lInput = self.mTrend.mSignalFrame;
        lOutput = self.forecast(lInput ,  self.mTimeInfo.mHorizon);
        return lOutput

    def plotResidues(self, name = None, format = 'png'):
        df = self.getForecastDatasetForPlots();
        lTime = self.mTimeInfo.mTime; # NormalizedTimeColumn;
        lPrefix = self.mSignal + "_";
        lPrefix2 = str(self.mOriginalSignal) + "_";
        if(name is not None):
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Trend' , lPrefix + 'Trend_residue', name = name + "_trend" , format=format);
            tsplot.decomp_plot(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue', name = name + "_cycle" , format=format);
            tsplot.decomp_plot(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue', name = name + "_AR" , format=format);
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'TransformedForecast' , lPrefix + 'TransformedResidue', name = name + "_transformed_forecast" , format=format);
            tsplot.decomp_plot(df, lTime, self.mOriginalSignal, lPrefix2 + 'Forecast' , lPrefix2 + 'Residue', name = name + "_forecast" , format=format);
        else:
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'Trend' , lPrefix + 'Trend_residue');
            tsplot.decomp_plot(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue');
            tsplot.decomp_plot(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue');
            tsplot.decomp_plot(df, lTime, self.mSignal, lPrefix + 'TransformedForecast' , lPrefix + 'TransformedResidue');
            tsplot.decomp_plot(df, lTime, self.mOriginalSignal, lPrefix2 + 'Forecast' , lPrefix2 + 'Residue');
        
    def standardPlots(self, name = None, format = 'png'):
        self.plotResidues(name = name, format=format);
        lInput = self.mTrend.mSignalFrame;
        lOutput = self.forecast(lInput ,  self.mTimeInfo.mHorizon);
        # print(lOutput.columns)
        lPrefix = str(self.mOriginalSignal) + "_";
        lForecastColumn = lPrefix + 'Forecast';
        lTime = self.mTimeInfo.mTime;            
        lOutput.set_index(lTime, inplace=True, drop=False);
        # print(lOutput[lTime].dtype);
        tsplot.prediction_interval_plot(lOutput,
                                        lTime, self.mOriginalSignal,
                                        lForecastColumn  ,
                                        lForecastColumn + '_Lower_Bound',
                                        lForecastColumn + '_Upper_Bound',
                                        name = name,
                                        format= format);
        #lOutput.plot()
        
    def getPlotsAsDict(self):
        lDict = {};
        df = self.getForecastDatasetForPlots();
        lTime = self.mTime;
        lSignalColumn = self.mOriginalSignal;
        lPrefix = self.mSignal + "_";
        lPrefix2 = str(self.mOriginalSignal) + "_";
        lDict["Trend"] = tsplot.decomp_plot_as_png_base64(df, lTime, self.mSignal, lPrefix + 'Trend' , lPrefix + 'Trend_residue', name = "trend");
        lDict["Cycle"] = tsplot.decomp_plot_as_png_base64(df, lTime, lPrefix + 'Trend_residue' , lPrefix + 'Cycle', lPrefix + 'Cycle_residue', name = "cycle");
        lDict["AR"] = tsplot.decomp_plot_as_png_base64(df, lTime, lPrefix + 'Cycle_residue' , lPrefix + 'AR' , lPrefix + 'AR_residue', name = "AR");
        lDict["Forecast"] = tsplot.decomp_plot_as_png_base64(df, lTime, lSignalColumn, lPrefix2 + 'Forecast' , lPrefix2 + 'Residue', name = "forecast");

        lInput = self.mModelFrame[[self.mTime, self.mOriginalSignal]];
        lOutput = self.forecast(lInput ,  self.mTimeInfo.mHorizon);
        # print(lOutput.columns)
        lPrefix = str(self.mOriginalSignal) + "_";
        lForecastColumn = lPrefix + 'Forecast';
        lTime = self.mTimeInfo.mTime;
        lOutput.set_index(lTime, inplace=True, drop=False);
        lDict["Prediction_Intervals"] = tsplot.prediction_interval_plot_as_png_base64(lOutput,
                                                                                      lTime, self.mOriginalSignal,
                                                                                      lForecastColumn  ,
                                                                                      lForecastColumn + '_Lower_Bound',
                                                                                      lForecastColumn + '_Upper_Bound',
                                                                                      name = "prediction_intervals");
        return lDict;


    def getVersions(self):
        
        import os, platform, pyaf
        lVersionDict = {};
        lVersionDict["PyAF_version"] = pyaf.__version__;
        lVersionDict["system_platform"] = platform.platform();
        lVersionDict["system_uname"] = platform.uname();
        lVersionDict["system_processor"] = platform.processor();
        lVersionDict["python_implementation"] = platform.python_implementation();
        lVersionDict["python_version"] = platform.python_version();

        import sklearn
        lVersionDict["sklearn_version"] = sklearn.__version__;

        import pandas as pd
        lVersionDict["pandas_version"] = pd.__version__;
    
        import numpy as np
        lVersionDict["numpy_version"] = np.__version__;
    
        import scipy as sc
        lVersionDict["scipy_version"] = sc.__version__;

        import matplotlib
        lVersionDict["matplotlib_version"] = matplotlib.__version__

        import pydot
        lVersionDict["pydot_version"] = pydot.__version__

        import sqlalchemy
        lVersionDict["sqlalchemy_version"] = sqlalchemy.__version__

        # print([(k, lVersionDict[k]) for k in sorted(lVersionDict)]);
        return lVersionDict;
    
