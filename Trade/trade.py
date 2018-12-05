# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:55:02 2018

@author: DIY
"""
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
import pandas as pd
import math
import EMD
import matplotlib.pyplot as plt
import trend
import anti_trend


time = pd.read_csv('e:\\EMD\\IF000.csv').iloc[:,0]
close = pd.read_csv("e:\\EMD\\IF000.csv").iloc[:,4]

#for i in range(len(Time)):
    #Time[i] = datetime.strptime(Time[i],"%Y/%m/%d %H:%M")

temp_time = np.array(time) 
Temp_time = map(lambda t:datetime.strptime(t,"%Y/%m/%d %H:%M"),temp_time)
Time = np.array(list(Temp_time))
 
###### 以开盘后90分钟为界进行信噪比检测 ######
###### start到end是计算指标的时间段
    ####### end的下一分钟是买入时间
def Start_End(tIndex):     ##### tIndex是起始Index
    """time是timestamp类型"""
    time = Time[tIndex]
    e = time + timedelta(hours=4.5)
    start = tIndex
    end = False
    for i in range(tIndex,len(Time)):
        if e == Time[i]:
            end = i
            break
    return start,end    ##### 返回为Index


def Trading_Time(tIndex):       ###### 返回Index
    temp = False
    if tIndex == False:
        #temp = False
        pass
    else:
        start,end = Start_End(tIndex)
        if end != False:
            EndT = Time[end]
            Buy_time = EndT+timedelta(seconds=60)
            if Buy_time.day != EndT.day:
                pass
                #return False
            else:
                for i in range(end,len(Time)):
                    if Time[i] == Buy_time and Time[i+1].day == Buy_time.day:
                        #print(i)
                        temp = i
                    else:
                        pass
    return temp


def ClosingTime(tIndex):        ####### 返回Index
    SIn = Trading_Time(tIndex)
    temp = False
    if SIn == False:
        pass
    else:
        #CIndex = False
        if SIn != False:
            for i in range(SIn,len(Time)):
                if Time[i].day == Time[SIn].day:
                    if len(Time) == i+1:
                        temp = i
                        break
                    elif Time[i+1].day != Time[SIn].day:
                        temp = i
                        break
    return temp
        
    
def Find_NextTime(start):   ####寻找下一个交易日起始Index
    """time是timestamp类型"""
    st = Time[start]
    st_day = st.day
    NIndex = False
    for j in range(start,len(Time)):
        if Time[j].day != st_day:
            NIndex = j
            break
    return NIndex
        

############### Trading ##############
RMS_In = 0
RME_In = 5399
NextIndex = 5400
#R_mean = EMD.Compute_R_mean(RMS_In,RME_In)

####### 交易账户
init_account = 4000000
fee = 0.0002
amount = 100
R = []
rf = 0.04
rev = 0
Max_Occupy = 0
value = 4000000
MV = []
LOGMV = []
MV.append(1)
LOGMV.append(0)


R_NT = []
R_MT = []
continuity = []
t = []
t.append(0)

while(NextIndex!=False):
    start,end = Start_End(NextIndex)
    Trade_S = Trading_Time(start)
    Trade_E = ClosingTime(start)
    if start!=False and end !=False and Trade_S!=False and Trade_E!=False:
        R_mean = EMD.Compute_R_mean(RMS_In,RME_In)
        R_now1 = EMD.compute_R(start,end)
        
        price_s = close[Trade_S]
        price_e = close[Trade_E]

        ## trend
        if (R_mean-R_now1)/abs(R_mean)>0.25 and R_now1<=-0.5:
            order = trend.trend_order(start+25,start+72,end)
            if order is not None:
                if order['type'] == 'short':
                    rev = (price_s-price_e)*amount*order['num']-fee*(price_e+price_s)*amount*order['num']
                    MV.append(MV[-1]*(1+(price_s*(1-fee)-price_e*(1+fee))/(price_s*(1+fee))))
                    print(str(Time[NextIndex])+"the open price is %f and close price is %f" %(price_s,price_e))
                    LOGMV.append(math.log(MV[-1]))
                    print('the MV is %s and the LOGMV is %s' %(MV[-1],LOGMV[-1]))
                    print("trend short rev: ",rev)
                    R.append(rev)
                    t.append(temp_time[start])
                    
                elif order['type'] == 'long':
                    rev = (price_e-price_s)*amount*order['num'] - fee*(price_e+price_s)*amount*order['num']
                
                    MV.append(MV[-1]*price_e*(1-fee)/(price_s*(1+fee)))
                    print(str(Time[NextIndex])+"the open price is %f and close price is %f" %(price_s,price_e))
                    LOGMV.append(math.log(MV[-1]))
                    print('the MV is %s and the LOGMV is %s' %(MV[-1],LOGMV[-1]))
                    print("trend long rev: ",rev)
                    R.append(rev)
                    t.append(temp_time[start])
                    
            elif order is None:
                print(str(Time[NextIndex])+" t close")
                MV.append(MV[-1])
                LOGMV.append(LOGMV[-1])
                t.append(temp_time[start])
                    
        
        ## anti-trend
        elif R_now1>=1.9*R_mean and R_mean>0:
            order = anti_trend.anti_trend_order(start+25,start+72,end)
            if order is not None:
                if order['type'] == 'short':
                    rev = (price_s-price_e)*amount*order['num']-fee*(price_e+price_s)*amount*order['num']
                    MV.append(MV[-1]*(1+(price_s*(1-fee)-price_e*(1+fee))/(price_s*(1+fee))))
                    print(str(Time[NextIndex])+"the open price is %f and close price is %f" %(price_s,price_e))
                    LOGMV.append(math.log(MV[-1]))
                    print('the MV is %s and the LOGMV is %s' %(MV[-1],LOGMV[-1]))
                    print("anti-trend short rev: ",rev)
                    R.append(rev)
                    t.append(temp_time[start])
                    
                elif order['type'] == 'long':
                    rev = (price_e-price_s)*amount*order['num'] - fee*(price_e+price_s)*amount*order['num']
                
                    MV.append(MV[-1]*price_e*(1-fee)/(price_s*(1+fee)))
                    print(str(Time[NextIndex])+"the open price is %f and close price is %f" %(price_s,price_e))
                    LOGMV.append(math.log(MV[-1]))
                    print('the MV is %s and the LOGMV is %s' %(MV[-1],LOGMV[-1]))
                    print("anti-trend long rev: ",rev)  
                    R.append(rev)
                    t.append(temp_time[start])
                    
            elif order is None:
                print(str(Time[NextIndex])+" a-t close")
                MV.append(MV[-1])
                LOGMV.append(LOGMV[-1])
                t.append(temp_time[start])
                
        else:
            print(str(Time[NextIndex])+" close")
            MV.append(MV[-1])
            LOGMV.append(LOGMV[-1])
            t.append(temp_time[start])
    
        R_NT.append(R_now1)
        R_MT.append(R_mean)
        if (price_e>price_s and close[end]>close[start+15]) or (price_s>price_e and close[start+15]>close[end]):
            continuity.append(1)
        else:
            continuity.append(0)
    
    
    NextIndex = Find_NextTime(start)
    RME_In = NextIndex - 1
    RMS_In = RME_In - 5000
    

df = pd.DataFrame()
df['R_NT'] = R_NT
df['R_MT'] = R_MT
df['continuity'] = continuity
df.to_csv('continue.csv')


####### 指标评估 #######
#sumReturn = np.sum(R)/init_account
#annualizedFactor = 245/((len(Time)-2175)/242)

#annualReturn = round(sumReturn*annualizedFactor,7)

#len(R)/2.83
#cum_Return = []
#sum = 0
#for i in R:
  #  sum += i
 #   cum_Return.append(sum)

#money = []
#for k in cum_Return:
 #   money.append(40000+k)

#rate = []
#for i in range(len(money)-1):
 #   rate.append(money[i+1]/money[i]-1)    

#volatility=np.std(rate,ddof=1)
#annual_volatility = volatility*math.sqrt(245)
#annual_sharpe_Ratio=(annualReturn-rf)/annual_volatility    



#max_drawdown =0
#for e, i in enumerate(money):
   # for f, j in enumerate(money):
  #      if f > e and float(j - i)  < max_drawdown:
 #           max_drawdown = float(j - i)

#max_drawdownratio =0
#try:
    #for e, i in enumerate(money):
   #     for f, j in enumerate(money):
  #          if f > e and float((j - i)/i)  < max_drawdownratio:
 #               max_drawdownratio = float((j - i)/i)
#except:
 #   max_drawdownratio=None


#win = 0
#for i in R:
 #   if i > 0:
  #      win += 1
#win_rate = win/float(len(R))


#plt.figure(figsize=(10, 5))
#summary
#print('Return')
#print(R)
#print('win_rate','annualReturn','annual_sharpe_Ratio','annual_volatility','max_drawdown','max_drawdownratio')
#print(win_rate,annualReturn,annual_sharpe_Ratio,annual_volatility,max_drawdown,max_drawdownratio)


plt.plot(t,MV)
plt.xlabel('Date')
plt.ylabel('MV')
plt.title('MV Curve')
plt.grid(True)
plt.show()        



plt.plot(R_NT)
plt.plot(R_MT)
plt.show()