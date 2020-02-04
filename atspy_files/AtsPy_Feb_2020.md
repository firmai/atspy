

```
!pip install atspy
```

    Collecting atspy
    [?25l  Downloading https://files.pythonhosted.org/packages/b2/a6/3fc563f6b52eef2226ff9e33acdddc5a282979df963c280b2e08e3ffd4d9/atspy-0.0.9.tar.gz (54kB)
    [K     |████████████████████████████████| 61kB 5.0MB/s 
    [?25hRequirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from atspy) (0.25.3)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from atspy) (1.4.1)
    Requirement already satisfied: numba in /usr/local/lib/python3.6/dist-packages (from atspy) (0.47.0)
    Collecting datetime
    [?25l  Downloading https://files.pythonhosted.org/packages/73/22/a5297f3a1f92468cc737f8ce7ba6e5f245fcfafeae810ba37bd1039ea01c/DateTime-4.3-py2.py3-none-any.whl (60kB)
    [K     |████████████████████████████████| 61kB 7.6MB/s 
    [?25hCollecting pmdarima
    [?25l  Downloading https://files.pythonhosted.org/packages/4a/1a/32945c19306212fd08547369f40c8965bbc9e18652bb241766dfab398710/pmdarima-1.5.2-cp36-cp36m-manylinux1_x86_64.whl (1.4MB)
    [K     |████████████████████████████████| 1.5MB 20.9MB/s 
    [?25hRequirement already satisfied: pydot in /usr/local/lib/python3.6/dist-packages (from atspy) (1.3.0)
    Requirement already satisfied: dill in /usr/local/lib/python3.6/dist-packages (from atspy) (0.3.1.1)
    Collecting pathos
    [?25l  Downloading https://files.pythonhosted.org/packages/c6/a2/cd59f73d5ede4f122687bf8f63de5d671c9561e493ca699241f05b038278/pathos-0.2.5.tar.gz (162kB)
    [K     |████████████████████████████████| 163kB 52.4MB/s 
    [?25hRequirement already satisfied: sqlalchemy in /usr/local/lib/python3.6/dist-packages (from atspy) (1.3.13)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from atspy) (3.1.2)
    Requirement already satisfied: xgboost in /usr/local/lib/python3.6/dist-packages (from atspy) (0.90)
    Requirement already satisfied: sklearn in /usr/local/lib/python3.6/dist-packages (from atspy) (0.0)
    Collecting mxnet==1.4.1
    [?25l  Downloading https://files.pythonhosted.org/packages/58/f4/bc147a1ba7175f9890523ff8f1a928a43ac8a79d5897a067158cac4d092f/mxnet-1.4.1-py2.py3-none-manylinux1_x86_64.whl (28.4MB)
    [K     |████████████████████████████████| 28.4MB 106kB/s 
    [?25hCollecting gluonts
    [?25l  Downloading https://files.pythonhosted.org/packages/98/c8/113009b077ca127308470dcd4851e53a6b4ad905fe61f36f28d22ff3a4a5/gluonts-0.4.2-py3-none-any.whl (323kB)
    [K     |████████████████████████████████| 327kB 63.5MB/s 
    [?25hCollecting nbeats-pytorch
      Downloading https://files.pythonhosted.org/packages/5c/b1/8718eb745b993d00f176d736e3aeb6850aa97eb6e279ff13493f9e4f3b93/nbeats_pytorch-1.3.0-py3-none-any.whl
    Collecting seasonal
      Downloading https://files.pythonhosted.org/packages/74/ec/1d6053a5bb05bf72a720772575b85a0a1599314e1ca2d6eba9000d75f4b8/seasonal-0.3.1-py2.py3-none-any.whl
    Collecting tbats
    [?25l  Downloading https://files.pythonhosted.org/packages/5e/f2/04545b598030cd72807847d9230a5db619658be3a650e112ed18acb3a122/tbats-1.0.9-py3-none-any.whl (43kB)
    [K     |████████████████████████████████| 51kB 7.8MB/s 
    [?25hCollecting tsfresh
    [?25l  Downloading https://files.pythonhosted.org/packages/1e/9a/5ecbfc08b555a706b463ccdd1d215c419a087222b57f31886293fd8a9697/tsfresh-0.13.0-py2.py3-none-any.whl (78kB)
    [K     |████████████████████████████████| 81kB 13.8MB/s 
    [?25hCollecting python-dateutil==2.8.0
    [?25l  Downloading https://files.pythonhosted.org/packages/41/17/c62faccbfbd163c7f57f3844689e3a78bae1f403648a6afb1d0866d87fbb/python_dateutil-2.8.0-py2.py3-none-any.whl (226kB)
    [K     |████████████████████████████████| 235kB 76.7MB/s 
    [?25hCollecting numpy==1.17.4
    [?25l  Downloading https://files.pythonhosted.org/packages/d2/ab/43e678759326f728de861edbef34b8e2ad1b1490505f20e0d1f0716c3bf4/numpy-1.17.4-cp36-cp36m-manylinux1_x86_64.whl (20.0MB)
    [K     |████████████████████████████████| 20.0MB 1.2MB/s 
    [?25hRequirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas->atspy) (2018.9)
    Requirement already satisfied: llvmlite>=0.31.0dev0 in /usr/local/lib/python3.6/dist-packages (from numba->atspy) (0.31.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from numba->atspy) (45.1.0)
    Collecting zope.interface
    [?25l  Downloading https://files.pythonhosted.org/packages/05/16/79fe71428c91673194a21fedcc46f7f1349db799bc2a65da4ffdbe570343/zope.interface-4.7.1-cp36-cp36m-manylinux2010_x86_64.whl (168kB)
    [K     |████████████████████████████████| 174kB 65.9MB/s 
    [?25hRequirement already satisfied: statsmodels>=0.10.0 in /usr/local/lib/python3.6/dist-packages (from pmdarima->atspy) (0.10.2)
    Requirement already satisfied: scikit-learn>=0.19 in /usr/local/lib/python3.6/dist-packages (from pmdarima->atspy) (0.22.1)
    Requirement already satisfied: Cython>=0.29 in /usr/local/lib/python3.6/dist-packages (from pmdarima->atspy) (0.29.14)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from pmdarima->atspy) (0.14.1)
    Requirement already satisfied: pyparsing>=2.1.4 in /usr/local/lib/python3.6/dist-packages (from pydot->atspy) (2.4.6)
    Collecting ppft>=1.6.6.1
    [?25l  Downloading https://files.pythonhosted.org/packages/5b/16/9e83c2aa45949ee9cd6e8731275acdaeb6c624b8728d6598196c65074f3e/ppft-1.6.6.1.tar.gz (63kB)
    [K     |████████████████████████████████| 71kB 13.1MB/s 
    [?25hCollecting pox>=0.2.7
    [?25l  Downloading https://files.pythonhosted.org/packages/6c/9a/957818485aa165ce93b646aeb20181215bb415c9dca8345bdbe23b08ae67/pox-0.2.7.tar.gz (110kB)
    [K     |████████████████████████████████| 112kB 66.9MB/s 
    [?25hRequirement already satisfied: multiprocess>=0.70.9 in /usr/local/lib/python3.6/dist-packages (from pathos->atspy) (0.70.9)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->atspy) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->atspy) (1.1.0)
    Requirement already satisfied: requests>=2.20.0 in /usr/local/lib/python3.6/dist-packages (from mxnet==1.4.1->atspy) (2.21.0)
    Collecting graphviz<0.9.0,>=0.8.1
      Downloading https://files.pythonhosted.org/packages/53/39/4ab213673844e0c004bed8a0781a0721a3f6bb23eb8854ee75c236428892/graphviz-0.8.4-py2.py3-none-any.whl
    Requirement already satisfied: tqdm~=4.23 in /usr/local/lib/python3.6/dist-packages (from gluonts->atspy) (4.28.1)
    Collecting ujson~=1.35
    [?25l  Downloading https://files.pythonhosted.org/packages/16/c4/79f3409bc710559015464e5f49b9879430d8f87498ecdc335899732e5377/ujson-1.35.tar.gz (192kB)
    [K     |████████████████████████████████| 194kB 65.2MB/s 
    [?25hRequirement already satisfied: boto3~=1.0 in /usr/local/lib/python3.6/dist-packages (from gluonts->atspy) (1.11.9)
    Collecting pydantic~=1.1
    [?25l  Downloading https://files.pythonhosted.org/packages/be/9a/a2d9613a70051615a84df6e9d697aad9787ba978bdeb4ad46c754457b3e1/pydantic-1.4-cp36-cp36m-manylinux2010_x86_64.whl (7.5MB)
    [K     |████████████████████████████████| 7.5MB 58.1MB/s 
    [?25hRequirement already satisfied: holidays<0.10,>=0.9 in /usr/local/lib/python3.6/dist-packages (from gluonts->atspy) (0.9.12)
    Requirement already satisfied: torch in /usr/local/lib/python3.6/dist-packages (from nbeats-pytorch->atspy) (1.4.0)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.6/dist-packages (from nbeats-pytorch->atspy) (0.5.0)
    Requirement already satisfied: patsy>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from tsfresh->atspy) (0.5.1)
    Requirement already satisfied: distributed>=1.18.3 in /usr/local/lib/python3.6/dist-packages (from tsfresh->atspy) (1.25.3)
    Requirement already satisfied: dask>=0.15.2 in /usr/local/lib/python3.6/dist-packages (from tsfresh->atspy) (2.9.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil==2.8.0->atspy) (1.12.0)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests>=2.20.0->mxnet==1.4.1->atspy) (1.24.3)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests>=2.20.0->mxnet==1.4.1->atspy) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests>=2.20.0->mxnet==1.4.1->atspy) (2019.11.28)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests>=2.20.0->mxnet==1.4.1->atspy) (2.8)
    Requirement already satisfied: botocore<1.15.0,>=1.14.9 in /usr/local/lib/python3.6/dist-packages (from boto3~=1.0->gluonts->atspy) (1.14.9)
    Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /usr/local/lib/python3.6/dist-packages (from boto3~=1.0->gluonts->atspy) (0.3.2)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3~=1.0->gluonts->atspy) (0.9.4)
    Requirement already satisfied: dataclasses>=0.6; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from pydantic~=1.1->gluonts->atspy) (0.7)
    Requirement already satisfied: pillow>=4.1.1 in /usr/local/lib/python3.6/dist-packages (from torchvision->nbeats-pytorch->atspy) (6.2.2)
    Requirement already satisfied: click>=6.6 in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (7.0)
    Requirement already satisfied: psutil>=5.0 in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (5.4.8)
    Requirement already satisfied: toolz>=0.7.4 in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (0.10.0)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (3.13)
    Requirement already satisfied: cloudpickle>=0.2.2 in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (1.2.2)
    Requirement already satisfied: sortedcontainers!=2.0.0,!=2.0.1 in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (2.1.0)
    Requirement already satisfied: zict>=0.1.3 in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (1.0.0)
    Requirement already satisfied: tornado>=4.5.1 in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (4.5.3)
    Requirement already satisfied: tblib in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (1.6.0)
    Requirement already satisfied: msgpack in /usr/local/lib/python3.6/dist-packages (from distributed>=1.18.3->tsfresh->atspy) (0.5.6)
    Requirement already satisfied: docutils<0.16,>=0.10 in /usr/local/lib/python3.6/dist-packages (from botocore<1.15.0,>=1.14.9->boto3~=1.0->gluonts->atspy) (0.15.2)
    Requirement already satisfied: heapdict in /usr/local/lib/python3.6/dist-packages (from zict>=0.1.3->distributed>=1.18.3->tsfresh->atspy) (1.0.1)
    Building wheels for collected packages: atspy, pathos, ppft, pox, ujson
      Building wheel for atspy (setup.py) ... [?25l[?25hdone
      Created wheel for atspy: filename=atspy-0.0.9-cp36-none-any.whl size=67575 sha256=1035faa251c802c39059296b5e41d1bf149e43603a58bf466ff349ae5eff38d8
      Stored in directory: /root/.cache/pip/wheels/31/0a/6f/85f4b8336a4613f1a0ccc4e93626be658b1d1ac7ef083cd30c
      Building wheel for pathos (setup.py) ... [?25l[?25hdone
      Created wheel for pathos: filename=pathos-0.2.5-cp36-none-any.whl size=77577 sha256=83c0d3aa0dfeaf09461669eeefed46acd5e6503f2460a287982aee865b010427
      Stored in directory: /root/.cache/pip/wheels/61/6d/83/90b0c3d2c271da2c4850731e894798c98f8dbedbac74e8eff0
      Building wheel for ppft (setup.py) ... [?25l[?25hdone
      Created wheel for ppft: filename=ppft-1.6.6.1-cp36-none-any.whl size=64708 sha256=158d97d2c02372f818ba1cb0fe717e64ac640cf545b6d373414ecbe5e050c5e9
      Stored in directory: /root/.cache/pip/wheels/6a/0c/53/ea8dd63608f75c1e7a64d5c5ce5d1e6d04f15ae8a6fce4c2a0
      Building wheel for pox (setup.py) ... [?25l[?25hdone
      Created wheel for pox: filename=pox-0.2.7-cp36-none-any.whl size=28303 sha256=8a2857548a54625375679ec5b752b0089ef343abd4bb9985532abb7a6c27772e
      Stored in directory: /root/.cache/pip/wheels/47/10/7b/0e916f6246fe7cf7d3acd25a6d273ecf3f97888cea073a8ac2
      Building wheel for ujson (setup.py) ... [?25l[?25hdone
      Created wheel for ujson: filename=ujson-1.35-cp36-cp36m-linux_x86_64.whl size=68035 sha256=593cce67833729edbe0ca0bd6d4f6f9d45bac203c035f47efd9c7bcfe1af85a6
      Stored in directory: /root/.cache/pip/wheels/28/77/e4/0311145b9c2e2f01470e744855131f9e34d6919687550f87d1
    Successfully built atspy pathos ppft pox ujson
    [31mERROR: datascience 0.10.6 has requirement folium==0.2.1, but you'll have folium 0.8.3 which is incompatible.[0m
    [31mERROR: albumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.9 which is incompatible.[0m
    [31mERROR: mxnet 1.4.1 has requirement numpy<1.15.0,>=1.8.2, but you'll have numpy 1.17.4 which is incompatible.[0m
    [31mERROR: nbeats-pytorch 1.3.0 has requirement numpy==1.16.2, but you'll have numpy 1.17.4 which is incompatible.[0m
    Installing collected packages: zope.interface, datetime, numpy, pmdarima, ppft, pox, pathos, graphviz, mxnet, ujson, pydantic, python-dateutil, gluonts, nbeats-pytorch, seasonal, tbats, tsfresh, atspy
      Found existing installation: numpy 1.17.5
        Uninstalling numpy-1.17.5:
          Successfully uninstalled numpy-1.17.5
      Found existing installation: graphviz 0.10.1
        Uninstalling graphviz-0.10.1:
          Successfully uninstalled graphviz-0.10.1
      Found existing installation: python-dateutil 2.6.1
        Uninstalling python-dateutil-2.6.1:
          Successfully uninstalled python-dateutil-2.6.1
    Successfully installed atspy-0.0.9 datetime-4.3 gluonts-0.4.2 graphviz-0.8.4 mxnet-1.4.1 nbeats-pytorch-1.3.0 numpy-1.17.4 pathos-0.2.5 pmdarima-1.5.2 pox-0.2.7 ppft-1.6.6.1 pydantic-1.4 python-dateutil-2.8.0 seasonal-0.3.1 tbats-1.0.9 tsfresh-0.13.0 ujson-1.35 zope.interface-4.7.1





```
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/firmai/random-assets-two/master/ts/monthly-beer-australia.csv")
df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Megaliters</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1956-01-01</th>
      <td>93.2</td>
    </tr>
    <tr>
      <th>1956-02-01</th>
      <td>96.0</td>
    </tr>
    <tr>
      <th>1956-03-01</th>
      <td>95.2</td>
    </tr>
    <tr>
      <th>1956-04-01</th>
      <td>77.1</td>
    </tr>
    <tr>
      <th>1956-05-01</th>
      <td>70.9</td>
    </tr>
  </tbody>
</table>
</div>




```
from atspy import AutomatedModel
```

    INFO:root:Using CPU



```
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
```


```
model_list=["HWAMS","HWAAS","TBAT"]

am = AutomatedModel(df = df , model_list=model_list, season="infer_from_data",forecast_len=60 )
forecast_in, performance = am.forecast_insample()
forecast_out = am.forecast_outsample()
all_ensemble_in, all_ensemble_out, all_performance = am.ensemble(forecast_in, forecast_out)
```

    <class 'pandas.core.frame.DataFrame'>
    The data has been successfully parsed by infering a frequency, and establishing a 'Date' index and 'Target' column.
    357
    An insample split of training size 357 and testing size 119 has been constructed
    Model HWAMS is being trained for in sample prediction
    Model HWAAS is being trained for in sample prediction
    Model TBAT is being trained for in sample prediction
    Model HWAMS is being used to forcast in sample
    Model HWAAS is being used to forcast in sample
    Model TBAT is being used to forcast in sample
    Successfully finished in sample forecast
    <class 'pandas.core.frame.DataFrame'>
    The data has been successfully parsed by infering a frequency, and establishing a 'Date' index and 'Target' column.
    Model HWAMS is being trained for out of sample prediction
    Model HWAAS is being trained for out of sample prediction
    Model TBAT is being trained for out of sample prediction
    Model HWAMS is being used to forcast out of sample
    Model HWAAS is being used to forcast out of sample
    Model TBAT is being used to forcast out of sample
    Successfully finished out of sample forecast
    Building LightGBM Ensemble from TS data (ensemble_lgb)
    Building LightGBM Ensemble from PCA reduced TSFresh Features (ensemble_ts). This can take a long time.
    LightGBM ensemble have been successfully built


    INFO:numexpr.utils:NumExpr defaulting to 2 threads.
    WARNING:tsfresh.feature_selection.relevance:Infered regression as machine learning task


    305  variables are found to be almost constant


    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {1.0, -1.0} in feature ''HWAAS__agg_linear_trend__f_agg_"max"__chunk_len_10__attr_"rvalue"''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {1.0, -1.0} in feature ''HWAAS__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"rvalue"''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.09531017980432521, 0.030716580297483365} in feature ''HWAAS__approximate_entropy__m_2__r_0.3''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.0, 180.0} in feature ''HWAAS__fft_coefficient__coeff_6__attr_"angle"''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.0, 2.0} in feature ''HWAAS__augmented_dickey_fuller__attr_"usedlag"''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.16666666666666666, 0.08333333333333333} in feature ''HWAAS__index_mass_quantile__q_0.1''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {1.0, 2.0} in feature ''HWAAS__number_peaks__n_1''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.4166666666666667, 0.5} in feature ''HWAAS__index_mass_quantile__q_0.4''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {2.0, 3.0} in feature ''HWAAS__number_cwt_peaks__n_1''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.5833333333333334, 0.6666666666666666} in feature ''HWAAS__index_mass_quantile__q_0.6''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.3333333333333333, 0.4166666666666667} in feature ''HWAAS__index_mass_quantile__q_0.3''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {1.0, -1.0} in feature ''HWAMS__agg_linear_trend__f_agg_"max"__chunk_len_10__attr_"rvalue"''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.6666666666666666, 0.75} in feature ''HWAAS__index_mass_quantile__q_0.7''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.9166666666666666, 1.0} in feature ''HWAAS__index_mass_quantile__q_0.9''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.25, 0.16666666666666666} in feature ''HWAAS__index_mass_quantile__q_0.2''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.3333333333333333, 0.25} in feature ''HWAAS__ratio_beyond_r_sigma__r_1''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.0, 180.0} in feature ''HWAMS__fft_coefficient__coeff_6__attr_"angle"''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {1.0, -1.0} in feature ''HWAMS__agg_linear_trend__f_agg_"min"__chunk_len_10__attr_"rvalue"''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.0, 2.0} in feature ''HWAMS__augmented_dickey_fuller__attr_"usedlag"''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {1.0, -1.0} in feature ''HWAMS__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"rvalue"''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.3333333333333333, 0.4166666666666667} in feature ''HWAMS__index_mass_quantile__q_0.3''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.25, 0.16666666666666666} in feature ''HWAMS__index_mass_quantile__q_0.2''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.4166666666666667, 0.5} in feature ''HWAMS__index_mass_quantile__q_0.4''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.5833333333333334, 0.6666666666666666} in feature ''HWAMS__index_mass_quantile__q_0.6''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.9166666666666666, 1.0} in feature ''HWAMS__index_mass_quantile__q_0.9''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.6666666666666666, 0.75} in feature ''HWAMS__index_mass_quantile__q_0.7''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {2.0, 3.0} in feature ''HWAMS__number_cwt_peaks__n_1''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.16666666666666666, 0.08333333333333333} in feature ''HWAMS__index_mass_quantile__q_0.1''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {0.08333333333333333, 0.16666666666666666} in feature ''HWAMS__ratio_beyond_r_sigma__r_1.5''.
    WARNING:tsfresh.feature_selection.significance_tests:[target_binary_feature_binary_test] A binary feature should have only values 1 and 0 (incl. True and False). Instead found {2.0, 3.0} in feature ''HWAMS__number_peaks__n_1''.
    WARNING:tsfresh.utilities.dataframe_functions:The columns ['HWAAS__agg_linear_trend__f_agg_"max"__chunk_len_50__attr_"intercept"'
     'HWAAS__agg_linear_trend__f_agg_"max"__chunk_len_50__attr_"rvalue"'
     'HWAAS__agg_linear_trend__f_agg_"max"__chunk_len_50__attr_"slope"' ...
     'TBAT__fft_coefficient__coeff_9__attr_"imag"'
     'TBAT__fft_coefficient__coeff_9__attr_"real"'
     'TBAT__spkt_welch_density__coeff_8'] did not have any finite values. Filling with zeros.


    LightGBM ensemble have been successfully built
    Building Standard First Level Ensemble
    Building Final Multi-level Ensemble



```
forecast_in.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>HWAMS</th>
      <th>HWAAS</th>
      <th>TBAT</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1985-10-01</th>
      <td>181.6</td>
      <td>161.962148</td>
      <td>162.391653</td>
      <td>148.410071</td>
    </tr>
    <tr>
      <th>1985-11-01</th>
      <td>182.0</td>
      <td>174.688055</td>
      <td>173.191756</td>
      <td>147.999237</td>
    </tr>
    <tr>
      <th>1985-12-01</th>
      <td>190.0</td>
      <td>189.728744</td>
      <td>187.649575</td>
      <td>147.589541</td>
    </tr>
    <tr>
      <th>1986-01-01</th>
      <td>161.2</td>
      <td>155.077205</td>
      <td>154.817215</td>
      <td>147.180980</td>
    </tr>
    <tr>
      <th>1986-02-01</th>
      <td>155.5</td>
      <td>148.054292</td>
      <td>147.477692</td>
      <td>146.773549</td>
    </tr>
  </tbody>
</table>
</div>




```
forecast_out.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HWAMS</th>
      <th>HWAAS</th>
      <th>TBAT</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1995-09-01</th>
      <td>137.518755</td>
      <td>137.133938</td>
      <td>142.906275</td>
    </tr>
    <tr>
      <th>1995-10-01</th>
      <td>164.136220</td>
      <td>165.079612</td>
      <td>142.865575</td>
    </tr>
    <tr>
      <th>1995-11-01</th>
      <td>178.671684</td>
      <td>180.009560</td>
      <td>142.827110</td>
    </tr>
    <tr>
      <th>1995-12-01</th>
      <td>184.175954</td>
      <td>185.715043</td>
      <td>142.790757</td>
    </tr>
    <tr>
      <th>1996-01-01</th>
      <td>147.166448</td>
      <td>147.440026</td>
      <td>142.756399</td>
    </tr>
  </tbody>
</table>
</div>




```
performance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>HWAMS</th>
      <th>HWAAS</th>
      <th>TBAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rmse</th>
      <td>0.000000</td>
      <td>17.599400</td>
      <td>18.993827</td>
      <td>36.538009</td>
    </tr>
    <tr>
      <th>mse</th>
      <td>0.000000</td>
      <td>309.738878</td>
      <td>360.765452</td>
      <td>1335.026136</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>155.293277</td>
      <td>142.399639</td>
      <td>140.577496</td>
      <td>126.590412</td>
    </tr>
  </tbody>
</table>
</div>




```
all_performance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rmse</th>
      <th>mse</th>
      <th>mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ensemble_lgb__X__HWAMS</th>
      <td>9.697588</td>
      <td>94.043213</td>
      <td>146.719412</td>
    </tr>
    <tr>
      <th>ensemble_lgb__X__HWAMS__X__HWAMS_HWAAS__X__ensemble_ts__X__HWAAS</th>
      <td>9.875212</td>
      <td>97.519817</td>
      <td>145.250837</td>
    </tr>
    <tr>
      <th>ensemble_lgb__X__HWAMS__X__HWAMS_HWAAS</th>
      <td>11.127326</td>
      <td>123.817378</td>
      <td>142.994374</td>
    </tr>
    <tr>
      <th>ensemble_lgb</th>
      <td>12.748526</td>
      <td>162.524907</td>
      <td>156.487208</td>
    </tr>
    <tr>
      <th>ensemble_lgb__X__HWAMS__X__HWAMS_HWAAS__X__ensemble_ts__X__HWAAS__X__HWAMS_HWAAS_TBAT__X__TBAT</th>
      <td>14.589155</td>
      <td>212.843442</td>
      <td>138.615567</td>
    </tr>
    <tr>
      <th>HWAMS</th>
      <td>15.567905</td>
      <td>242.359663</td>
      <td>136.951615</td>
    </tr>
    <tr>
      <th>HWAMS_HWAAS</th>
      <td>16.651370</td>
      <td>277.268110</td>
      <td>135.544299</td>
    </tr>
    <tr>
      <th>ensemble_ts</th>
      <td>17.255107</td>
      <td>297.738716</td>
      <td>163.134079</td>
    </tr>
    <tr>
      <th>HWAAS</th>
      <td>17.804066</td>
      <td>316.984751</td>
      <td>134.136983</td>
    </tr>
    <tr>
      <th>HWAMS_HWAAS_TBAT</th>
      <td>23.358758</td>
      <td>545.631579</td>
      <td>128.785846</td>
    </tr>
    <tr>
      <th>TBAT</th>
      <td>39.003864</td>
      <td>1521.301380</td>
      <td>115.268940</td>
    </tr>
  </tbody>
</table>
</div>




```
all_ensemble_in[["Target","ensemble_lgb__X__HWAMS","HWAMS","HWAAS"]].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f42f01de470>




![png](AtsPy_Feb_2020_files/AtsPy_Feb_2020_9_1.png)



```
all_ensemble_out[["ensemble_lgb__X__HWAMS","HWAMS","HWAAS"]].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f42d40747b8>




![png](AtsPy_Feb_2020_files/AtsPy_Feb_2020_10_1.png)



```
am.models_dict_in
```




    {'HWAAS': <statsmodels.tsa.holtwinters.HoltWintersResultsWrapper at 0x7f42f7822d30>,
     'HWAMS': <statsmodels.tsa.holtwinters.HoltWintersResultsWrapper at 0x7f42f77fff60>,
     'TBAT': <tbats.tbats.Model.Model at 0x7f42d3aab048>}




```
am.models_dict_out
```




    {'HWAAS': <statsmodels.tsa.holtwinters.HoltWintersResultsWrapper at 0x7f9c01309278>,
     'HWAMS': <statsmodels.tsa.holtwinters.HoltWintersResultsWrapper at 0x7f9c01309cf8>,
     'TBAT': <tbats.tbats.Model.Model at 0x7f9c08f18ba8>}


