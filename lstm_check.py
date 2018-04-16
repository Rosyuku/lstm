#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:01:27 2017

@author: kazuyuki
"""

import pandas as pd

if __name__ == "__main__":
    
    log = 'result_1/log'
    
    df = pd.read_json(log)
    
    df.index = df['iteration']
    
    df['main/loss'].plot(loglog=True)
    df['main/loss'].rolling(10).median().plot(loglog=True)
    df['main/loss'].rolling(100).median().plot(loglog=True)