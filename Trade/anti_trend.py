# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:41:47 2018

接受trade中产生的anti-trend的信号
@author: DIY
"""

import numpy as np
import talib
import pandas as pd

openPrice = pd.read_csv("e:\\EMD\\IF000.csv").iloc[:,1]
closePrice = pd.read_csv("e:\\EMD\\IF000.csv").iloc[:,4]
high = pd.read_csv("e:\\EMD\\IF000.csv").iloc[:,2]
low = pd.read_csv("e:\\EMD\\IF000.csv").iloc[:,3]

#flag = 0  ## 1表示多头，-1表示空头
MA_Close = talib.MA(np.array(closePrice),timeperiod=5000)
MA_Low = talib.MA(np.array(low),timeperiod=5000)
MA_High = talib.MA(np.array(high),timeperiod=5000)


def antiTrend(start,end):
    H = max(high[start:end])
    L = min(low[start:end])
    C = closePrice[end]
    
  
    if min(closePrice[start:end])<np.mean(MA_High[start:end]) and C > MA_High[end]:
    #if C>r3:    
        flag = 1
    elif max(closePrice[start:end])>np.mean(MA_Low[start:end]) and C < MA_Low[end]:
    #elif C<rm2:
        flag = -1
    elif max(closePrice[start:end])>np.mean(MA_High[start:end]) and C < MA_Close[end]:
    #elif C<s3:  
        flag = -1
    elif min(closePrice[start:end])<np.mean(MA_Low[start:end]) and C > MA_Close[end]:
    #elif C>sm2:    
        flag = 1
    else:
        flag = 0
        
    return flag


def anti_trend_order(ini,start,end):
    order = None
    signal = antiTrend(ini,end)
    if signal == 0:
        return order
    elif signal == 1:
        order = {"type":'long',"num":10}
        return order
    elif signal == -1:
        order = {"type":'short',"num":10}
        return order
    