from .ssa import mySSA
from .nbeats import plot_scatter, data_generator, train_100_grad_steps, load, save, eval_test
from .pyaf import cForecastEngine as autof

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import seaborn as sns
from fbprophet import Prophet
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
import warnings
from matplotlib.pylab import rcParams
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split as tts
#import pyaf.ForecastEngine as autof
from datetime import timedelta
from gluonts.model.prophet import ProphetPredictor  
from dateutil.relativedelta import relativedelta
import torch
from torch import optim
from torch.nn import functional as F
from nbeats_pytorch.model import NBeatsNet # some import from the trainer script e.g. load/save functions.

pd.plotting.register_matplotlib_converters()
warnings.filterwarnings("ignore")

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/path/to/application/app/folder')



###### Utility Functions

def infer_seasonality(train):
  ssa = mySSA(train)
  ssa.embed(embedding_dimension=36, verbose=True)
  ssa.decompose(True)
  rec = ssa.view_reconstruction(ssa.Xs[1], names="Seasonality", return_df=True,plot=False)
  peaks, _ = find_peaks(rec.values.reshape(len(rec),), height=0)
  peak_diffs = [j-i for i, j in zip(peaks[:-1], peaks[1:])]
  seasonality = max(peak_diffs,key=peak_diffs.count)
  return seasonality 

def infer_periodocity(train):
  if pd.infer_freq(df.index)=="MS":
    periodocity = 12
  return periodocity 

def select_seasonality(train, season):
  if season == "periodocity":
    seasonality = infer_periodocity(train)
  elif season== "infer_from_data":
    seasonality = infer_seasonality(train)
  return seasonality


def add_freq(idx, freq=None):
    """Add a frequency attribute to idx, through inference or directly.

    Returns a copy.  If `freq` is None, it is inferred.
    """
    idx = idx.copy()
    if freq is None:
        if idx.freq is None:
            freq = pd.infer_freq(idx)
        else:
            return idx
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError('no discernible frequency found to `idx`.  Specify'
                             ' a frequency string with `freq`.')
    return idx
    
def parse_data(df):
  if type(df) ==pd.DataFrame:
    if df.shape[1]>1:
      raise ValueError("The dataframe should only contain one target column")
  elif type(df) == pd.Series:
    df = df.to_frame()
  else:
    raise TypeError("Please supply a pandas dataframe with one column or a pandas series")
  try:
    df.index.date
  except AttributeError: 
    raise TypeError("The index should be a datetype")
  print(type(df))
  if df.isnull().any().values[0]:
    raise ValueError("The dataframe cannot have any null values, please interpolate")
  try:
    df.columns = ["Target"]
  except:
    raise ValueError("There should only be one column")

  df.index = df.index.rename("Date")
  df.index = add_freq(df.index)

  return df, pd.infer_freq(df.index)

def gluonts_dataframe(df):
  freqed = pd.infer_freq(df.index)
  if freqed=="MS":
    freq= "M"
    #start = df.index[0] + relativedelta(months=1)
  else:
    freq= freqed
  df = ListDataset(
    [{"start": df.index[0], "target": df.values}],
    freq =freq )
  return df

def nbeats_dataframe(df, forecast_length, in_sample, train_portion=0.75):

  backcast_length = 1 * forecast_length

  df = df
  print(df.head())
  df = df.values  # just keep np array here for simplicity.
  norm_constant = np.max(df)
  df = df / norm_constant  # small leak to the test set here.

  x_train_batch, y = [], []
  
  for i in range(backcast_length+1, len(df) - forecast_length +1): #25% to 75% so 50% #### Watch out I had to plus one. 
      x_train_batch.append(df[i - backcast_length:i])
      y.append(df[i:i + forecast_length])

  x_train_batch = np.array(x_train_batch)[..., 0]
  y = np.array(y)[..., 0]

  #x_train, y_train = x_train_batch, y

  net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
        forecast_length=forecast_length,
        thetas_dims=[7, 8],
        nb_blocks_per_stack=3,
        backcast_length=backcast_length,
        hidden_layer_units=128,
        share_weights_in_stack=False,
        device=device)

  if in_sample==True:
    c = int(len(x_train_batch) * train_portion)  
    x_train, y_train = x_train_batch[:c], y[:c]
    x_test, y_test = x_train_batch[c:], y[c:]

    return x_train, y_train, x_test, y_test, net, norm_constant
  
  else:
    c = int(len(x_train_batch) * 1)  
    x_train, y_train = x_train_batch[:c], y[:c]
    
    return x_train, y_train, net, norm_constant


def train_test_split(df, train_proportion=0.75):
  print(df)

  size = int(df['Target'].shape[0]*train_proportion); print(size)
  train, test = tts(df['Target'], train_size=size,shuffle=False, stratify=None)
  return train, test

## Dictionary Parameters - Makes sense topic area. 
## In this case it would be very easy to make an app available online. 
## This would allow you to play around with APIs and that sort of thing. 

def prophet_dataframe(df):
  df_pr = df.reset_index()
  df_pr.columns = ['ds','y']
  return df_pr

def original_dataframe(df, freq):
  prophet_pred = pd.DataFrame({"Date" : df['ds'], "Target" : df["yhat"]})
  prophet_pred = prophet_pred.set_index("Date")
  #prophet_pred.index.freq = pd.tseries.frequencies.to_offset(freq)
  return prophet_pred["Target"]

device = torch.device('cpu')  # use the trainer.py to run on GPU.
CHECKPOINT_NAME = 'nbeats-training-checkpoint.th'

def train_models(train, models,forecast_len, full_df=None,seasonality="infer_from_data",in_sample=None):

  seasons = select_seasonality(train, seasonality)

  models_dict = {}
  for m in models:
    if m=="ARIMA":
      models_dict["ARIMA"] = pm.auto_arima(train, seasonal=True, m=seasons)
    if m=="Prophet":
      model = Prophet()
      models_dict["Prophet"] = model.fit(prophet_dataframe(train))
    if m=="HWAAS":
      models_dict["HWAAS"] = ExponentialSmoothing(train, seasonal_periods=seasons, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
    if m=="HWAMS":
      models_dict["HWAMS"] = ExponentialSmoothing(train, seasonal_periods=seasons, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
    # if m=="HOLT":
    #   models_dict["HOLT"] = Holt(train,exponential=True).fit()
    if m=="PYAF":
      model = autof.cForecastEngine()
      model.train(iInputDS = train.reset_index(), iTime = 'Date', iSignal = 'Target', iHorizon = len(train)) # bad coding to have horison here
      models_dict["PYAF"] = model.forecast(iInputDS = train.reset_index(), iHorizon = forecast_len)
    if m=="Gluonts":
      freqed = pd.infer_freq(train.index)
      if freqed=="MS":
        freq= "M"
      else:
        freq= freqed
      estimator = DeepAREstimator(freq=freq, prediction_length=forecast_len, trainer=Trainer(epochs=2)) #use_feat_dynamic_real=True
      print(train)
      print(type(train))
      print(gluonts_dataframe(train))
      models_dict["Gluonts"] = estimator.train(training_data=gluonts_dataframe(train)) 
    if m=="NBEATS":

      device = torch.device('cpu')
      seasons = select_seasonality(train,seasonality)

      if os.path.isfile(CHECKPOINT_NAME):
          os.remove(CHECKPOINT_NAME)
      stepped = 5
      batch_size = 10
      if in_sample:
        x_train, y_train, x_test, y_test, net, norm_constant = nbeats_dataframe(full_df, forecast_len, in_sample=True)
        optimiser = optim.Adam(net.parameters())
        data = data_generator(x_train, y_train, batch_size)
        #test_losses = []
        for r in range(stepped):
   
            train_100_grad_steps(data, device, net, optimiser) #test_losses
        models_dict["NBEATS"] = {}
        models_dict["NBEATS"]["model"] = net
        models_dict["NBEATS"]["x_test"] = x_test
        models_dict["NBEATS"]["y_test"] = y_test
        models_dict["NBEATS"]["constant"] = norm_constant

      else: # if out_sample train is df

        x_train, y_train,net, norm_constant= nbeats_dataframe(full_df, forecast_len, in_sample=False)

        batch_size = 10  # greater than 4 for viz
        optimiser = optim.Adam(net.parameters())
        data = data_generator(x_train, y_train, batch_size)
        stepped = 5
        #test_losses = []
        for r in range(stepped):
            _, forecast = net(torch.tensor(x_train, dtype=torch.float)) ### Not Used
            p = forecast.detach().numpy()                               ### Not Used
            train_100_grad_steps(data, device, net, optimiser) #test_losses
        models_dict["NBEATS"] = {}
        models_dict["NBEATS"]["model"] = net
        models_dict["NBEATS"]["tuple"] = (x_train, y_train,net, norm_constant)


    # if m=="ProphetGluonts":
    #   freqed = pd.infer_freq(train.index)
    #   if freqed=="MS":
    #     freq= "M"
    #   else:
    #     freq= freqed
    #   models_dict["ProphetGluonts"] = ProphetPredictor(freq=freq, prediction_length=forecast_len) #use_feat_dynamic_real=True
    #   models_dict["ProphetGluonts"] = list(models_dict["ProphetGluonts"])

# create a forecast engine. This is the main object handling all the operations
# We use the test-dataset as the last step of our training to generate the evaluation-metrics and do not use the test-dataset during prediction.
# get the best time series model for predicting one week

  return models_dict

def forecast_models(models_dict, forecast_len, freq, df, in_sample=True): # test here means any df
  forecast_dict = {}
  for name, model in models_dict.items():
    if name=="ARIMA":
      forecast_dict[name] = model.predict(forecast_len)
    if name=="Prophet":
      future = model.make_future_dataframe(periods=forecast_len,freq=freq)
      future_pred = model.predict(future)
      print(future_pred)
      print(future_pred.shape)
      print(original_dataframe(future_pred,freq)[-forecast_len:])
      print(original_dataframe(future_pred,freq))
      print(original_dataframe(future_pred,freq).shape)
      forecast_dict[name] = original_dataframe(future_pred,freq)[-forecast_len:]
    if name=="HWAAS":
      forecast_dict[name] = model.forecast(forecast_len)
    if name=="HWAMS":
      forecast_dict[name] = model.forecast(forecast_len)
    # if name=="HOLT":
    #   forecast_dict[name] = model.forecast(forecast_len)
    if name=="PYAF":
      print(model["Target_Forecast"][-forecast_len:])
      print(model["Target_Forecast"][-forecast_len:].shape)

      forecast_dict[name] = model["Target_Forecast"][-forecast_len:].values
    if name=="Gluonts":
      if freq=="MS":
        freq= "M"
        if in_sample:
          for df_entry, forecast in zip(gluonts_dataframe(df), model.predict(gluonts_dataframe(df))):
            forecast_dict[name] = forecast.samples.mean(axis=0)
        else:
          future = ListDataset([{"target": df[-forecast_len:], "start": df.index[-1] + relativedelta(months=1)}],freq=freq)
          #future = ListDataset([{"target": [df[-1]]*forecast_len, "start": df.index[-1] + relativedelta(months=1)}],freq=freq)          
          
          for df_entry, forecast in zip(future, model.predict(future)): #next(predictor.predict(future))
            forecast_dict[name] = forecast.samples.mean(axis=0) # .quantile(0.5)
    if name=="NBEATS":
      if in_sample:
        net = model["model"]
        x_test = model["x_test"]
        y_test = model["y_test"]
        norm_constant =  model["constant"]
        net.eval()
        _, forecast = net(torch.tensor(x_test, dtype=torch.float))
        p = forecast.detach().numpy()
        forecast_dict[name] = p[-1]*norm_constant
      else:
        net = model["model"]
        net.eval()
        x_train, y_train,net, norm_constant= model["tuple"]
        _, forecast = net(torch.tensor(x_train, dtype=torch.float))
        p = forecast.detach().numpy()
        forecast_dict[name] = p[-1]*norm_constant


  return forecast_dict

def forecast_frame(test, forecast_dict):
  insample = test.to_frame()
  for name, forecast in forecast_dict.items():
    insample[name] = forecast
  return insample

def forecast_frame_insample(forecast_dict,test):
    insample = test.to_frame()
    for name, forecast in forecast_dict.items():
      insample[name] = forecast
    return insample 

def forecast_frame_outsample(forecast_dict,df,forecast_len):
    #can be done in the future, but too hard to date here. They can do it.  
    #outsample = df.to_frame().iloc[-:forecast_len,:]
    #outsample.index = pd.date_range(outsample.index[0]+ timedelta(days=1),outsample.index[0]+ timedelta(days=forecast_len) )
    #make_future_dataframe(periods=forecast_len,freq=freq)
    ra = -1
    for name, forecast in forecast_dict.items():
      ra += 1
      if ra==0:
        outsample = pd.DataFrame(forecast,columns=[name])
      else:
        outsample[name] = forecast
    return outsample 


def insample_performance(test, forecast_dict,dict=False):
  forecasts = forecast_frame(test, forecast_dict)
  dict_perf = {}
  for col, values in forecasts.iteritems():
    dict_perf[col] = {}
    dict_perf[col]["rmse"] = rmse(forecasts["Target"], forecasts[col])
    dict_perf[col]["mse"] = dict_perf[col]["rmse"]**2
    dict_perf[col]["mean"] = forecasts[col].mean()
  if dict:
    return dict_perf
  else:
    return pd.DataFrame.from_dict(dict_perf)

# def weighted_model()

#   for name, forecast in forecast_dict.items():
#     insample[name] = forecast

#### Class Functions





###### CLASS CREATION


# frozen=True
@dataclass()
class AutomatedModel():
    """A configuration for the Menu.

    Attributes:
        title: The title of the Menu.
        body: The body of the Menu.
        button_text: The text for the button label.
        cancellable: Can it be cancelled?
    """

    df: pd.Series
    #model_list: list = ["ARIMA","HOLT"]
    model_list: list 
    season: str = "infer_from_data"
    forecast_len: int = 20

    def train_insample(self):
      dataframe, freq = parse_data(self.df)
      train, test = train_test_split(dataframe, train_proportion=0.75)
      forecast_len = len(test)
      models = train_models(train, models= self.model_list,forecast_len= forecast_len,full_df=dataframe,seasonality=self.season,in_sample=True )
      return models, freq, test
    
    def train_outsample(self):
      dataframe, freq = parse_data(self.df)
      models = train_models(dataframe["Target"], models= self.model_list,forecast_len = self.forecast_len,full_df=dataframe, seasonality=self.season,in_sample=False)
      return models, freq, dataframe["Target"]

    def forecast_insample(self):
      models_dict, freq, test = self.train_insample()
      forecast_len = test.shape[0]
      forecast_dict = forecast_models(models_dict, forecast_len, freq,test, in_sample=True)
      forecast_frame = forecast_frame_insample(forecast_dict,test)
      
      preformance = insample_performance(test, forecast_frame)
      return forecast_frame, preformance

    def forecast_outsample(self):
      models_dict, freq, dataframe  = self.train_outsample()
      forecast_dict = forecast_models(models_dict, self.forecast_len, freq,dataframe,in_sample=False)
      forecast_frame = forecast_frame_outsample(forecast_dict,self.df,self.forecast_len)
      return forecast_frame