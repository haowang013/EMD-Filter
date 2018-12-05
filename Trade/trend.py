# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:42:05 2018

接受trade产生的trend信号，产生order
@author: DIY
"""
import pandas as pd
import numpy as np
import talib

openPrice = pd.read_csv("e:\\EMD\\IF000.csv").iloc[:,1]
closePrice = pd.read_csv("e:\\EMD\\IF000.csv").iloc[:,4]
high = pd.read_csv("e:\\EMD\\IF000.csv").iloc[:,2]
low = pd.read_csv("e:\\EMD\\IF000.csv").iloc[:,3]
MOM = talib.MOM(np.array(closePrice), timeperiod=150)
MA_MOM = talib.MA(np.array(MOM),timeperiod=900)

MA_low = talib.MA(np.array(low),timeperiod=1000)
MA_high = talib.MA(np.array(high),timeperiod=1000)

def trend_order(ini,start,end):
    highest = max(closePrice[ini:end])
    lowest = min(closePrice[ini:end])
    pt = closePrice[end]
    p00 = closePrice[ini]
    p0 = closePrice[start]
    
    for i,p in enumerate(closePrice[ini:end]):
        if p == highest:
            temp_h = i
        elif p == lowest:
            temp_l = i
            
    if p0>p00 and (pt-p0)>0 and pt>max(MA_high[ini:end]) and MOM[end]>MA_MOM[end] and temp_h>temp_l:
        return {"type":'long',"num":10}
    elif p0<p00 and (pt-p0)<0 and pt<min(MA_low[ini:end]) and MOM[end]<MA_MOM[end] and temp_h<temp_l:
        return {"type":'short',"num":10}
    else:
        return None