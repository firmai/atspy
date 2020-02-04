# Automated Time Series Models in Python (AtsPy)

Easily develop sophisticated state of the art time series models to forecast the future values of a data series based on its historical patterns. More than ten advanced time series prediction models are developed to give the best possible prediction performance. This package gives you the ability to identify which model fits your data the best. You can solve problems like sales forecasting, production estimation, product demand, staff optimisation, inventory management and more. 

#### Install
```
pip install atspy
```

#### Load
```python
from atspy import AutomatedModel
```

#### Automated Models

1. ARIMA - Automated ARIMA Modelling
1. Prophet - Modeling Multiple Seasonality With Linear or Non-linear Growth
1. HWAAS - Exponential Smoothing With Additive Trend and Additive Seasonality
1. HWAMS - Exponential Smoothing with Additive Trend and Multiplicative Seasonality
1. PYAF - Feature Generating Model (slow and underforms)
1. NBEATS -  Neural basis expansion analysis (now fixed at 20 Epochs)
1. Gluonts - RNN-based Model (now fixed at 20 Epochs)
1. TATS - Seasonal and Trend no Box Cox
1. TBAT - Trend and Box Cox
1. TBATS1 - Trend, Seasonal (one), and Box Cox
1. TBATP1 - TBATS1 but Seasonal Inference is Hardcoded by Periodicity
1. TBATS2 - TBATS1 With Two Seasonal Periods

#### Why AtsPy?

1. Implements all your favourite automated time series models in a unified manner by simply running ```AutomatedModel(df)```.
1. Automatically identifies the seasonalities in your data using Singular Spectrum Analysis and peak analysis with ```"infer_from_data"```.
1. Identifies and makes accessible the best model for your time series. 
1. Combines the predictions of all these models in a simple (average) or complex (GBM) ensemble for improved performance.

#### AtsPy Progress 

1. Univariate forecasting only (single column). 
1. Holt, ETS, ARIMA, Prophet, Gluonts, N-BEATS, and PYAF.
1. So far I have only tested monthly data, and only one particular dataseries. 
1. More work ahead; suggestions and criticisms appreciated, use issues tab.

#### AtsPy Future Development

1. The creation of signal processes features for further improvements in the ensemble prediction.
1. Additional in-sample validation steps to stop deep learning models from over and underfitting. 
1. The implementation of more automation models to further improve the ensemble prediction. 
1. A range of accessibility and tracking functions to improve the usability. 
1. Code annotations for other developers to follow the work being done. 
1. Remove some seasonality assumptions; keep seasonality stable between in and out of sample.  

### Documentation by Example

----------

#### Pandas DataFrame
```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/firmai/random-assets-two/master/ts/monthly-beer-australia.csv")
df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month")
```

#### AtsPy AutomatedModel
```python
from atspy import AutomatedModel
model_list = ["HWAMS","HWAAS","ARIMA","Gluonts","PYAF","Prophet","NBEATS"]
am = AutomatedModel(df = df , model_list=model_list, season="infer_from_data",forecast_len=20 )
forecast_in, performance = am.forecast_insample()
forecast_out = am.forecast_outsample()
```
### Example Output
![](https://github.com/firmai/atspy/blob/master/atspy_files/Screen%20Shot%202020-01-31%20at%207.51.07%20PM.png)

