# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:13:22 2018

@author: DIY
"""
import numpy as np
import pandas as pd
from datetime import datetime
import math
import matplotlib.pyplot as plt

def X_Index(X):
    Index = []
    for i in range(len(X)):
        Index.append(i)
    return Index
        
def Local_Max(X):
    Max_Index, L_Mx = [],[]
    Max_Index.append(0)
    L_Mx.append(X[0])
    for i in range(1,len(X)-1):
        if (X[i] > X[i-1] and X[i] > X[i+1]):
            Max_Index.append(i)
            L_Mx.append(X[i])
    Max_Index.append(len(X)-1)
    L_Mx.append(X[len(X)-1])
    return Max_Index,L_Mx

def Local_Min(X):
    Min_Index, L_mn = [], []
    Min_Index.append(0)
    L_mn.append(X[0])
    for i in range(1,len(X)-1):
        if (X[i] < X[i-1] and X[i] < X[i+1]):
            Min_Index.append(i)
            L_mn.append(X[i])
    Min_Index.append(len(X)-1)
    L_mn.append(X[len(X)-1])
    return Min_Index,L_mn

############### Cubic Spline Interpolation #############
###### 三对角矩阵分解 #######
def TDMA(X,Y):
    """len(A) = n-1; len(D) = n; len(B) = n; len(C) = n-1;
        [[0,A][B][C,0]]X = D"""
    A,B,C,D = [],[],[],[]  ## A存储距离h
    C.append(0)
    B.append(1)
    D.append(0)
    for i in range(len(X)-2):
        A.append(X[i+1]-X[i])
        C.append(X[i+2]-X[i+1])
        B.append(2*(X[i+2]-X[i]))
        D.append(6*((Y[i+2]-Y[i+1])/(X[i+2]-X[i+1]))-((Y[i+1]-Y[i])/(X[i+1]-X[i])))
    A.append(0)
    B.append(1)
    D.append(0)
    
    C_, D_, X = [],[],[]
    C_.append(C[0]/B[0])
    for i in range(1,len(C)):
        C_.append(C[i]/(B[i]-C_[i-1]*A[i-1]))
        
    D_.append(D[0]/B[0])
    for i in range(1,len(D)):
        D_.append((D[i]-D_[i-1]*A[i-1])/(B[i]-C_[i-1]*A[i-1]))
    
    ##### X暂时反向存放，eg X[0]储存X[n],X[1]储存X[n-1]    
    X.append(D_[-1])
    for i in range(len(C)-1,-1,-1):
        X.append(D_[i]-C_[i]*X[-1])
    ##### X转换排列顺序
    X.reverse()
    return X

####### 边界条件假设：自由边界，首尾两端没有受到任何弯曲的力 ########
    ########## S'' = 0 ----> m_0 = 0, m_n = 0 ##########
    ###### m_i = S''_i(x_i), h_i = x_i+1 - x_i为步长 ######
    ###### Y是序列的端点值 ######
def Cubic_S_Inter(Index,Y):
    """TDMA求得的X即为矩阵方程的二次微分值；
       Si(x) = ai + bi(x-xi)+ci(x-xi)^2+di(x-xi)^3;
       有0-n共n+1个数据点，则步长len(H)=n, 共n个方程"""
    M = TDMA(Index,Y)
    A,B,C,D = [],[],[],[]
    for i in range(len(Index)-1):
        h = Index[i+1] - Index[i]
        A.append(Y[i])
        B.append((Y[i+1]-Y[i])/h - h/2*M[i] - h/6*(M[i+1]-M[i]))
        C.append(M[i]/2)
        D.append((M[i+1]-M[i])/(6*h))
    A.append(Y[-1])
    coef = {"a":A,"b":B,"c":C,"d":D}
    return coef

def SPline(Index,Y):
    T = []
    Y_ = []
    coef = Cubic_S_Inter(Index,Y)
    for i in range(len(Index)-1):
        t = np.arange(Index[i],Index[i+1],0.5)
        x = Index[i]
        a = coef['a'][i]
        b = coef['b'][i]
        c = coef['c'][i]
        d = coef['d'][i]
        for j in t:
            y = a + b*(j-x) + c*math.pow((j-x),2) + d*math.pow((j-x),3)
            T.append(j)
            Y_.append(y)
    return T,Y_

def Spline_Func(x,MIndex,coef):
    #if x == MIndex[0]:
     #   a = coef['a'][0]
      #  b = coef['b'][0]
      #  c = coef['c'][0]
      #  d = coef['d'][0]
      #  y = a + b*(x-MIndex[0]) + c*math.pow((x-MIndex[0]),2) + d*math.pow((x-MIndex[0]),3)
      #  return y
    if MIndex[-1] == x:
        a = coef['a'][-1]
        y = a 
        return y
    else:
        for i in range(len(MIndex)-1):
            if MIndex[i] <= x and MIndex[i+1] > x:
                a = coef['a'][i]
                b = coef['b'][i]
                c = coef['c'][i]
                d = coef['d'][i]
                y = a + b*(x-MIndex[i]) + c*math.pow((x-MIndex[i]),2) + d*math.pow((x-MIndex[i]),3)
                #print(y)
                return y

def Minus_Ave(Max,MaxIndex,Min,MinIndex,Index,Y):
    coef_max = Cubic_S_Inter(MaxIndex,Max)
    coef_min = Cubic_S_Inter(MinIndex,Min)
    Y_ = []
    for i in range(len(Index)):
        x = Index[i]
        y_max = Spline_Func(x,MaxIndex,coef_max)
        #print(y_max)
        y_min = Spline_Func(x,MinIndex,coef_min)
        #print(y_min)
        Ave_m = 1/2*(y_max+y_min)
        #print(Ave)
        #print(type(Ave))
        Y_.append(Y[i]-Ave_m)
        #if i == len(Index)-1:
         #   print("YMax"+str(y_max)+"--"+"YMin"+str(y_min)+"--"+str(Y[i]))
    return Y_


def Compute_R_mean(start,end):
    ETF = 'e:\\EMD\\IF000.csv'
    X_R = np.array(pd.read_csv(ETF).iloc[start:end,4])
    ############ IMF1
    X = X_R
    ##### round1 #####
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
    #plt.plot(Index,h,Index,st)
    #plt.show()
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    
    ##### round2-5 #####
    for i in range(3):
        temp = h
        Index = X_Index(temp)
        MIndex,LM = Local_Max(temp)
        Min_Index,LMin = Local_Min(temp)
        T,Y = SPline(MIndex,LM)
        T2,Y2 = SPline(Min_Index,LMin)
        st = []
        for i in range(len(Index)):
            st.append(0)
        h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,temp)

    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    IMF1 = h
    X = X_R - IMF1
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
    
########### IMF2
    ##### round1 #####
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    ##### round2-5 #####
    for i in range(3):
        temp = h
        Index = X_Index(temp)
        MIndex,LM = Local_Max(temp)
        Min_Index,LMin = Local_Min(temp)
        T,Y = SPline(MIndex,LM)
        T2,Y2 = SPline(Min_Index,LMin)
        st = []
        for i in range(len(Index)):
            st.append(0)
        h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,temp)

    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    
    IMF2 = h
    X = X - IMF2
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
    
    
############# IMF3
    ##### round1 #####
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
    
    ##### round2-5 #####   
    for i in range(3):
        temp = h
        Index = X_Index(temp)
        MIndex,LM = Local_Max(temp)
        Min_Index,LMin = Local_Min(temp)
        T,Y = SPline(MIndex,LM)
        T2,Y2 = SPline(Min_Index,LMin)
        st = []
        for i in range(len(Index)):
            st.append(0)
        h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,temp)
        
    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    
    IMF3 = h
    X = X - IMF3
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
    
    
############ IMF4 
    ###### round 1 ######
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
        
    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    
    IMF4 = h
    X = X - IMF4
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
    
############### IMF5    
    ###### round 1 ######
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)

        
    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    
    IMF5 = h
    X = X - IMF5
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
    
################# IMF6
    ###### round 1 ######
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
        
    #plt.plot(Index,h,Index,st)
    #plt.show()  
    
    
    IMF6 = h
    X = X - IMF6
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
    
############### IMF7
    ###### round 1 ######
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
        
    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    
    IMF7 = h
    X = X - IMF7
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
############## IMF8
    ####### round 1
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)    
        
    #plt.plot(Index,h,Index,st)
    #plt.show()
    

    IMF8 = h
    X = X - IMF8
    #plt.plot(Index,X,Index,X_R)
    #plt.show()

    R_mean = np.log(np.std(X_R-X)/np.std(X))
    
    return R_mean


def compute_R(start,end):
    ETF = 'e:\\EMD\\IF000.csv'
    X_R = np.array(pd.read_csv(ETF).iloc[start:end,4])

############ IMF1
    X = X_R
    ##### round1 #####
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
    #plt.plot(Index,h,Index,st)
    #plt.show()
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    
    ##### round2-5 #####
    for i in range(3):
        temp = h
        Index = X_Index(temp)
        MIndex,LM = Local_Max(temp)
        Min_Index,LMin = Local_Min(temp)
        T,Y = SPline(MIndex,LM)
        T2,Y2 = SPline(Min_Index,LMin)
        st = []
        for i in range(len(Index)):
            st.append(0)
        h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,temp)

    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    IMF1 = h
    X = X_R - IMF1
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
    
########### IMF2
    ##### round1 #####
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    ##### round2-5 #####
    for i in range(3):
        temp = h
        Index = X_Index(temp)
        MIndex,LM = Local_Max(temp)
        Min_Index,LMin = Local_Min(temp)
        T,Y = SPline(MIndex,LM)
        T2,Y2 = SPline(Min_Index,LMin)
        st = []
        for i in range(len(Index)):
            st.append(0)
        h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,temp)

    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    
    IMF2 = h
    X = X - IMF2
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
        
############# IMF3
    ##### round1 #####
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
    
    ##### round2-5 #####   
    for i in range(3):
        temp = h
        Index = X_Index(temp)
        MIndex,LM = Local_Max(temp)
        Min_Index,LMin = Local_Min(temp)
        T,Y = SPline(MIndex,LM)
        T2,Y2 = SPline(Min_Index,LMin)
        st = []
        for i in range(len(Index)):
            st.append(0)
        h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,temp)
        
    #plt.plot(Index,h,Index,st)
    #plt.show()
     
    IMF3 = h
    X = X - IMF3
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
     
############### IMF4 
    ###### round 1 ######
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)
   
    #plt.plot(Index,h,Index,st)
    #plt.show()
     
    
    IMF4 = h
    X = X - IMF4
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
############### IMF5    
    ###### round 1 ######
    Index = X_Index(X)
    MIndex,LM = Local_Max(X)
    Min_Index,LMin = Local_Min(X)
    T,Y = SPline(MIndex,LM)
    T2,Y2 = SPline(Min_Index,LMin)
    #plt.plot(Index,X,T,Y,T2,Y2)
    #plt.show()
    st = []
    for i in range(len(Index)):
        st.append(0)
    h = Minus_Ave(LM,MIndex,LMin,Min_Index,Index,X)

        
    #plt.plot(Index,h,Index,st)
    #plt.show()
    
    
    IMF5 = h
    X = X - IMF5
    #plt.plot(Index,X,Index,X_R)
    #plt.show()
    
    R = np.log(np.std(X_R-X)/np.std(X))
    
    return R