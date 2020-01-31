# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import pandas as pd
import numpy as np
import itertools

from . import SignalHierarchy as sighier


class cSignalGrouping (sighier.cSignalHierarchy):

    def __init__(self):
        sighier.cSignalHierarchy.__init__(self)
        self.mLabels2Tuples = {};
        
    def tuple_to_string(self, k):
        str1 = "_".join(list(k));
        # print(k , "=>" , str1);
        return str1;
    
    def add_level(self, previous_level):
        level = previous_level + 1;
        self.mStructure[level] = {};
        for group in self.mStructure[previous_level]:
            lGroupLabel = group; # self.tuple_to_string(group);
            lTuple = self.mLabels2Tuples[lGroupLabel]
            for k in [previous_level]:
                if(lTuple[k] != ""):
                    new_group = list(lTuple);
                    new_group[k] = "";
                    new_group = tuple(new_group);
                    lNewGroupLabel = self.tuple_to_string(new_group);
                    self.mLabels2Tuples[lNewGroupLabel] = new_group;
                    if(lNewGroupLabel not in self.mStructure[level]):
                        self.mStructure[level][lNewGroupLabel] = set();
                    self.mStructure[level][lNewGroupLabel].add(lGroupLabel)
        # print("STRUCTURE_LEVEL" , level, self.mStructure[level]);

    def create_HierarchicalStructure(self):
        
        # lGroups = {};
        # lGroups["State"] = ["NSW","VIC","QLD","SA","WA","NT","ACT","TAS"];
        # lGroups["Gender"] = ["female","male"];
        # lHierarchy['GroupOrder']= ["State" , "Gender"];
        
        lGroups = self.mHierarchy['Groups']
        self.mLevels = lGroups.keys();
        self.mLabels2Tuples = {};
        self.mStructure = {};
        array1 = [ sorted(lGroups[k]) for k in self.mHierarchy['GroupOrder'] ];
        prod = itertools.product( *array1 );
        # print(prod);
        # prod = itertools.product(['a' , 'b'] , ['1' , '2'] , ['cc' , 'dd']);
        level = 0;
        self.mStructure[level] = {}
        for k in prod:
            # print("PRODUCT_DETAIL", k);
            lGroupLabel = self.tuple_to_string(k);
            # Grouping genrates all possible group combinations.
            # Not all columns are mandatory. 
            if(lGroupLabel in self.mTrainingDataset.columns):
                self.mLabels2Tuples[lGroupLabel] = k;
                self.mStructure[level][lGroupLabel] = set();
        # print("STRUCTURE_LEVEL" , level, self.mStructure[level]);
        while(len(self.mStructure[level]) > 1):
            self.add_level(level);
            level = level + 1;
        
        # print("STRUCTURE", self.mStructure);
        pass
