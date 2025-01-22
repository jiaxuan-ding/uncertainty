# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:59:47 2023

@author: jiaxu
"""

import numpy as np
import pandas as pd
import random as random
import math
from scipy.stats import multivariate_normal
import scipy as sp
from scipy.linalg import fractional_matrix_power


from pyarma import *
from scipy.stats import norm

########### load data  #################
path='entry your path address/'
data=pd.read_csv(path+'lab data.csv') 
datao=pd.read_csv(path+'online data.csv')

########### setting parameters##########
#lab data
Nid=len(data['ID'].unique()) #number of respondants
Nscenario=5 #each person complete 5 scenarios
#street-intercept data
Nido=len(datao['ID'].unique()) #number of respondants
Nscenarioo=2 #each person complete 2 scenarios
Nalter=2 #numner of alternative:3
Nattri=4#number of attributes：3

#attribute: RC, OC and DR
data['OC_1']=data['OC']
data.loc[(data['alt']==3),['OC','OC_1']]=data.loc[(data['alt']==3),['P1_V','P2_V']].to_numpy()
data[['OC','OC_1']]=data[['OC','OC_1']].astype('float64')
data['RC']=-((data['RC']-np.min(data['RC']))/(np.max(data['RC'])-np.min(data['RC'])))
data['OC']=-((data['OC']-np.min(data['OC']))/(np.max(data['OC'])-np.min(data['OC'])))
data['OC_1']=-((data['OC_1']-np.min(data['OC_1']))/(np.max(data['OC_1'])-np.min(data['OC_1'])))
data['DR']=np.log(data['DR'])
data['DR']=((data['DR']-np.min(data['DR']))/(np.max(data['DR'])-np.min(data['DR'])))
data_att=(data.loc[:,['RC','OC','OC_1','DR']].to_numpy()).reshape(Nid*Nscenario,Nalter,Nattri).astype('float64') #attribut:RC, OC, DR
datao['OC_1']=datao['OC']
datao.loc[(datao['alt']==3),['OC','OC_1']]=datao.loc[(datao['alt']==3),['P1_V','P2_V']].to_numpy()
datao[['OC','OC_1']]=datao[['OC','OC_1']].astype('float64')
datao['RC']=-((datao['RC']-np.min(datao['RC']))/(np.max(datao['RC'])-np.min(datao['RC'])))
datao['OC']=-((datao['OC']-np.min(datao['OC']))/(np.max(datao['OC'])-np.min(datao['OC'])))
datao['OC_1']=-((datao['OC_1']-np.min(datao['OC_1']))/(np.max(datao['OC_1'])-np.min(datao['OC_1'])))
datao['DR']=np.log(datao['DR'])
datao['DR']=((datao['DR']-np.min(datao['DR']))/(np.max(datao['DR'])-np.min(datao['DR'])))
datao_att=(datao.loc[:,['RC','OC','OC_1','DR']].to_numpy()).reshape(Nido*Nscenarioo,Nalter,Nattri).astype('float64') #attribut:RC, OC, DR
data_att=np.vstack((data_att,datao_att))
#reference level
refer=np.zeros((Nid*Nscenario+Nido*Nscenarioo,1,Nattri))
refer[:,0,0]=data_att[:,0,0]
refer[:,0,1]=-np.sum(data_att[:,1,1:3]*np.vstack((data.loc[(data['alt']==3),['P1','P2']].to_numpy(),datao.loc[(datao['alt']==3),['P1','P2']].to_numpy())),axis=1)
refer[:,0,2]=-np.sum(data_att[:,1,1:3]*np.vstack((data.loc[(data['alt']==3),['P1','P2']].to_numpy(),datao.loc[(datao['alt']==3),['P1','P2']].to_numpy())),axis=1)
refer[:,0,3]=data_att[:,0,3]
#state probability
p_weight=np.ones((Nid*Nscenario+Nido*Nscenarioo,Nalter,Nattri))
p_weight[:,1,1:3]=np.vstack((data.loc[(data['alt']==3),['P1','P2']].to_numpy(),datao.loc[(datao['alt']==3),['P1','P2']].to_numpy()))
#choice
y_choice=np.vstack((data[['chosen']].to_numpy(),datao[['chosen']].to_numpy())).reshape(Nid*Nscenario+Nido*Nscenarioo,Nalter)


###################define the LLH #########################
def cal_LL1(param,dataset,p_weight,refer,decoy_choice):
    afa=(param[1]);lamuda=(param[2]);ra=(param[3])
    beta=(np.array([param[4],param[5],param[5],param[6]]))
    
    datasetn=(dataset-refer)*beta
    datasetn[datasetn<0]=-lamuda*np.power((-datasetn[datasetn<0]),afa)
    datasetn[datasetn>=0]=np.power(datasetn[datasetn>=0],afa)
    
    use_ev=0
    if use_ev:
    #use EV as reference
        p_weightn=np.exp(-np.power(-np.log(p_weight),ra))
    else:
    #use ICEV as reference
        p_weightn=p_weight.copy()
        p_weightn[:,1,1]=np.exp(-np.power(-np.log(p_weight[:,1,1]),ra))
        p_weightn[:,1,2]=1-np.exp(-np.power(-np.log(p_weight[:,1,1]),ra))
    
    u=np.zeros((dataset.shape[0],2))
    u[:,0]=np.sum(datasetn[:,0,[0,1,3]]*p_weightn[:,0,[0,1,3]],axis=1)
    u[:,1]=np.sum(datasetn[:,1,:]*p_weightn[:,1,:]+param[0],axis=1)
    V=np.exp(u)
    P=V/np.sum(V,axis=1,keepdims=True)
    P[P<=0.00001]=0.00001
    MLL=decoy_choice*np.log(P)
    return -np.sum(MLL)

#####################estimation##########################
Nfeval = 1

def callbackF(Xi):
    global Nfeval
    print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}    {7: 3.6f}     {8: 3.6f}   '.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3],Xi[4], Xi[5],Xi[6],  cal_LL1(Xi, data_att, p_weight,refer,y_choice)))
    Nfeval += 1
 
print  ('{0}      {1}       {2}      {3}      {4}      {5}        {6}        {7}        {8}   '.format('Iter', ' asc_ev', ' afa',  ' lamuda', ' r', 'RC','OC','DR','-LLH'))     


bounds=((-np.Inf,np.Inf),(0,1),(1,np.Inf),(0,np.Inf),(0,np.Inf),(0,np.Inf),(0,np.Inf))

intial=np.array([2.4368652,  0.88858026, 3.92863228, 4.30796582, 1.15238074 ,4.89709058,
 4.18926677])


resOpt= sp.optimize.minimize(
            fun = cal_LL1,
            x0 = intial,
            args = ( data_att, p_weight,refer,y_choice),
            method ='L-BFGS-B' ,
            callback=callbackF,
            bounds=bounds,
            tol=0.01,
            options = {'disp': True}
            )

##################results#####################
#Translates a p-value into a significance level based on predefined thresholds.
def trans_significance(p_value):
    sig_level=[]
    for i in p_value:
        if i < 0.01:
            sig_level.append('***')
        elif i < 0.1:
            sig_level.append('**')
        elif i < 0.317:
            sig_level.append('*')
        else:
            sig_level.append(' ')
    return sig_level

EST=resOpt['x'].copy()#Estimtaed parameter
ERR=(np.sqrt(np.diag(resOpt.hess_inv.todense()))).copy() #Standard erro
Z_VAL=EST/ERR #z-value
P_VAL=2*norm.cdf(-np.abs(Z_VAL)) #p-value
SIG=trans_significance(P_VAL) #singificant level
        
###########################################return results########################################
info = f"""
Estimation summary for MCPT
------------------------------------------------------------------------------------
Coefficient           Estimate      Std.Err.       z-val         P>|z|
------------------------------------------------------------------------------------
asc_EV            {EST[0]:10.2f}   {ERR[0]:10.2f}     {Z_VAL[0]:10.2f}     {P_VAL[0]:10.2f} {SIG[0]}
\u03B2_RC              {EST[4]:10.2f}   {ERR[4]:10.2f}     {Z_VAL[4]:10.2f}     {P_VAL[4]:10.2f} {SIG[4]}
\u03B2_OC              {EST[5]:10.2f}   {ERR[5]:10.2f}     {Z_VAL[5]:10.2f}     {P_VAL[5]:10.2f} {SIG[5]}
\u03B2_DR              {EST[6]:10.2f}   {ERR[6]:10.2f}     {Z_VAL[6]:10.2f}     {P_VAL[6]:10.2f} {SIG[6]}
\u03B1                 {EST[1]:10.2f}   {ERR[1]:10.2f}     {Z_VAL[1]:10.2f}     {P_VAL[1]:10.2f} {SIG[1]}
\u03BB                 {EST[2]:10.2f}   {ERR[2]:10.2f}     {Z_VAL[2]:10.2f}     {P_VAL[2]:10.2f} {SIG[2]}
r                 {EST[3]:10.2f}   {ERR[3]:10.2f}     {Z_VAL[3]:10.2f}     {P_VAL[3]:10.2f} {SIG[3]}
------------------------------------------------------------------------------------
Significance:  0.01 '***' 0.1 '**' 0.317 '*' 

Log-Likelihood={-resOpt['fun']:10.2f}
BIC={len(intial)*np.log(Nid*Nscenario+Nido*Nscenarioo)+2*resOpt['fun']:10.2f}

"""

# Print the formatted information
print(info)


