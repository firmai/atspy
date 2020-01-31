# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np

class cSignalQuantizer:
    def __init__(self):
        pass


    def signal2quant(self, x , curve):
        return min(curve.keys(), key=lambda y:abs(float(curve[y])-x))
    
    def quant2signal(self, series , iSignal,  Q):
        return series.apply(lambda x : iSignal.quantile(x / Q))

    def quantizeSignal(self, iSignal , Q) :
        q = pd.Series(range(0,Q)).apply(lambda x : iSignal.quantile(x/Q))
        curve = q.to_dict()
        lSignal_Q = iSignal.apply(lambda x : self.signal2quant(x, curve)) + 1
        s = self.quant2signal(lSignal_Q , iSignal, Q)
        #return lSignal_Q
        return iSignal
