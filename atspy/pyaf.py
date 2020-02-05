# import atspy.TS.SignalDecomposition as tsdec
# import atspy.TS.Options as tsopts
# import atspy.TS.Perf as tsperf
# import atspy.TS.Utils as tsutil
# import atspy.TS.Utils as tsutil

# class cForecastEngine:
        
#     def __init__(self):
#         self.mSignalDecomposition = tsdec.cSignalDecomposition();
#         self.mOptions = tsopts.cSignalDecomposition_Options();
#         pass

    
#     def train(self , iInputDS, iTime, iSignal, iHorizon, iExogenousData = None):
#         try:
#             self.mSignalDecomposition.mOptions = self.mOptions;
#             self.mSignalDecomposition.train(iInputDS, iTime, iSignal, iHorizon, iExogenousData);
#         except tsutil.PyAF_Error as error:
#             raise error
#         except Exception as error:
#             # print('caught this training error: ' + repr(error))            
#             raise tsutil.PyAF_Error("TRAIN_FAILED");
#         pass

#     def forecast(self , iInputDS, iHorizon):
#         try:
#             lForecastFrame = self.mSignalDecomposition.forecast(iInputDS, iHorizon);
#             return lForecastFrame;
#         except tsutil.PyAF_Error as error:
#             raise error
#         except Exception as error:
#             # print('caught this forecast error: ' + repr(error))
#             raise tsutil.PyAF_Error("FORECAST_FAILED");
        
#     def getModelInfo(self):
#         return  self.mSignalDecomposition.getModelInfo();

    
#     def standardPlots(self , name = None, format = 'png'):
#         self.mSignalDecomposition.standardPlots(name, format);

#     def getPlotsAsDict(self):
#         return self.mSignalDecomposition.getPlotsAsDict();

#     def to_json(self):
#         return self.mSignalDecomposition.to_json();

#     def computePerf(self, actual, predicted , name):
#         lForecastPerf =  tsperf.cPerf();
#         lForecastPerf.compute(actual, predicted, name);
#         return lForecastPerf;

#     def generateCode(self, iDSN = None, iDialect = None):
#         from CodeGen import TS_CodeGen_Objects as tscodegen
#         lCodeGenerator = tscodegen.cDecompositionCodeGenObject(iDSN, iDialect);
#         lSQL = lCodeGenerator.generateCode(self);
#         # print("GENERATED_SQL_CODE" , lSQL);
#         return lSQL;