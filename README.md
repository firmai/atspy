# Automated Time Series (AtsPy)

#### Automating Automated Time Series Models
1. Univariate forecasting only (single column). 
1. Holt, ETS, ARIMA, Prophet, Gluonts, N-BEATS, and PYAF.
1. At some point a meta model will be tested to see if an ensemble would work. 
1. So far I have only tested monthly data, and only one particular dataseries. 
1. More work ahead; suggestions and criticisms appreciated, use issues tab.

#### Install
```
pip install atspy
```

#### First Step Load Pandas DataFrame
```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/firmai/random-assets-two/master/ts/monthly-beer-australia.csv")
df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month")
```

#### Second Decide on Models and Run.
```python
model_list = ["HWAMS","HWAAS","ARIMA","Gluonts","PYAF","Prophet","NBEATS"]
am = AutomatedModel(df = df , model_list=model_list, season="infer_from_data",forecast_len=20 )
forecast_in, performance = am.forecast_insample()
forecast_out = am.forecast_outsample()
```

![](https://github.com/firmai/atspy/blob/master/atspy_files/Screen%20Shot%202020-01-31%20at%207.51.07%20PM.png)

