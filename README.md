# Automated Time Series Models in Python (AtsPy)

Easily develop state of the art time series models to forecast the future values of a data series. Simply load your data and select which models you want to test. This is the largest repository of automated structural and machine learning time series models. Please get in contact if you want to contribute a model.  

#### Install
```
pip install atspy
```

#### Automated Models

1. ```ARIMA``` - Automated ARIMA Modelling
1. ```Prophet``` - Modeling Multiple Seasonality With Linear or Non-linear Growth
1. ```HWAAS``` - Exponential Smoothing With Additive Trend and Additive Seasonality
1. ```HWAMS``` - Exponential Smoothing with Additive Trend and Multiplicative Seasonality
1. ```PYAF``` - Feature Generating Model (slow and underforms)
1. ```NBEATS``` -  Neural basis expansion analysis (now fixed at 20 Epochs)
1. ```Gluonts``` - RNN-based Model (now fixed at 20 Epochs)
1. ```TATS``` - Seasonal and Trend no Box Cox
1. ```TBAT``` - Trend and Box Cox
1. ```TBATS1``` - Trend, Seasonal (one), and Box Cox
1. ```TBATP1``` - TBATS1 but Seasonal Inference is Hardcoded by Periodicity
1. ```TBATS2``` - TBATS1 With Two Seasonal Periods

#### Why AtsPy?

1. Implements all your favourite automated time series models in a unified manner by simply running ```AutomatedModel(df)```.
1. Reduce structural model errors with 30-50% by using LightGBM with TSFresh infused features.  
1. Automatically identify the seasonalities in your data using singular spectrum analysis, periodograms, and peak analysis.
1. Identifies and makes accessible the best model for your time series using in sample validation methods.  
1. Combines the predictions of all these models in a simple (average) and complex (GBM) ensembles for improved performance.
1. Where appropriate models have been developed to use GPU resources to speed up the automation process.
1. Easily access all the models by using ```am.models_dict_in``` for in sample and ```am.models_dict_out``` for out of sample.

#### AtsPy Progress 

1. Univariate forecasting only (single column) and only for monthly data (daily data will be available soon). 
1. So far I have only tested monthly data and only one particular dataseries. 
1. More work ahead; all suggestions and criticisms appreciated, use the issues tab.


### Documentation by Example

----------
#### Load Package
```python
from atspy import AutomatedModel
```

#### Pandas DataFrame
```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/firmai/random-assets-two/master/ts/monthly-beer-australia.csv")
df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month")
```

1. ```TATS``` - Seasonal and Trend no Box Cox
1. ```TBAT``` - Trend and Box Cox
1. ```TBATS1``` - Trend, Seasonal (one), and Box Cox
1. ```TBATP1``` - TBATS1 but Seasonal Inference is Hardcoded by Periodicity
1. ```TBATS2``` - TBATS1 With Two Seasonal Periods

#### AtsPy AutomatedModel
```python
from atspy import AutomatedModel
model_list = ["HWAMS","HWAAS","TBAT"]
am = AutomatedModel(df = df , model_list=model_list,forecast_len=20 )
```

Other models to try, add as many as you like, note ```ARIMA``` is slow: ```"ARIMA","Gluonts","PYAF","Prophet","NBEATS", "TATS", "TBATS1", "TBATP1", "TBATS2"```

```
forecast_in, performance = am.forecast_insample()
```
### Example Output
![](https://github.com/firmai/atspy/blob/master/atspy_files/Screen%20Shot%202020-01-31%20at%207.51.07%20PM.png)





#### AtsPy Future Development

1. Additional in-sample validation steps to stop deep learning models from over and underfitting. 
1. Code annotations for other developers to follow and improve on the work being done. 
1. Force seasonality stability between in and out of sample training models.
1. Make AtsPy less dependency heavy, currently it draws on tensorflow, pytorch and mxnet. 



