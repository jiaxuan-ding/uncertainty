# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:31:36 2024

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
path='enter your path here/'
data=pd.read_csv(path+'lab data.csv') 
datao=pd.read_csv(path+'online data.csv')

#######setting parameters##############
Nalter=2 #numner of alternative:2
Nattri=4#number of attributes：4
#scenario index
choice_set_index=np.array(['R1.1s', 'R1.1r', 'R1.2s', 'R1.2r', 'R1.3s', 'R1.3r', 'R1.4s','R1.4r', 'R1.5s', 'R1.5r', 'R2.1s', 'R2.1r', 'R2.2s', 'R2.2r','R2.3s', 'R2.3r', 'R2.4s', 'R2.4r', 'R2.5s', 'R2.5r'])
#lab data
Nid=len(data['ID'].unique()) #number of respondants
Nscenario=5 #each person complete 5 scenarios
#online data
Nido=len(datao['ID'].unique()) #number of respondants
Nscenarioo=2 #each person complete 2 scenarios
#attribute value
data['OC_1']=data['OC']
data.loc[(data['alt']==3),['OC','OC_1']]=data.loc[(data['alt']==3),['P1_V','P2_V']].to_numpy()
data[['OC','OC_1']]=data[['OC','OC_1']].astype('float64')
data_att=(data.loc[:,['RC','OC','OC_1','DR']].to_numpy()).reshape(100,Nscenarioo,Nalter,Nattri).astype('float64') #attribut:RC, OC, DR
datao['OC_1']=datao['OC']
datao.loc[(datao['alt']==3),['OC','OC_1']]=datao.loc[(datao['alt']==3),['P1_V','P2_V']].to_numpy()
datao[['OC','OC_1']]=datao[['OC','OC_1']].astype('float64')
datao_att1=(datao.loc[:,['RC','OC','OC_1','DR']].to_numpy()).reshape(Nido,Nscenarioo,Nalter,Nattri).astype('float64') #attribut:RC, OC, DR
data_att=np.vstack((data_att,datao_att1))
data_att[:,:,:,0]= (-(data_att[:,:,:,0]-np.min(data_att[:,:,:,0]))/(np.max(data_att[:,:,:,0])-np.min(data_att[:,:,:,0])) )  +0.01  
data_att[:,:,:,1]= (-(data_att[:,:,:,1]-np.min(data_att[:,:,:,1]))/(np.max(data_att[:,:,:,1])-np.min(data_att[:,:,:,1]))   )  +0.01  
data_att[:,:,:,2]= (-(data_att[:,:,:,2]-np.min(data_att[:,:,:,2]))/(np.max(data_att[:,:,:,2])-np.min(data_att[:,:,:,2]))   )  +0.01  
data_att[:,:,:,3]=np.log(data_att[:,:,:,3])
data_att[:,:,:,3]= ((data_att[:,:,:,3]-np.min(data_att[:,:,:,3]))/(np.max(data_att[:,:,:,3])-np.min(data_att[:,:,:,3]))   )  +0.01  
#choice
y_choice_1=(data.loc[(data['chosen']!=0),'alt'].to_numpy()).reshape(100,Nscenarioo,1)-1
y_choice_1[y_choice_1==2]=1
y_choice_1o=(datao.loc[(datao['chosen']!=0),'alt'].to_numpy()).reshape(Nido,Nscenarioo,1)-1
y_choice_1o[y_choice_1o==2]=1
y_choice_1=np.vstack((y_choice_1,y_choice_1o))

######DFT parameters##############
#equal attention weight
w_p=np.zeros(((Nido+100)*Nscenarioo,Nattri))
w_p[:,0]=1/3;w_p[0:Nid*Nscenario,1]=(1/3)*(data.loc[(data['alt']==3),['P1']].to_numpy()).flatten();w_p[0:Nid*Nscenario,2]=(1/3)*(data.loc[(data['alt']==3),['P2']].to_numpy()).flatten();w_p[:,3]=1/3
w_p[Nid*Nscenario:Nid*Nscenario+Nido*Nscenarioo,1]=(1/3)*(datao.loc[(datao['alt']==3),['P1']].to_numpy()).flatten();w_p[Nid*Nscenario:Nid*Nscenario+Nido*Nscenarioo,2]=(1/3)*(datao.loc[(datao['alt']==3),['P2']].to_numpy()).flatten()
w_p=w_p.reshape((Nido+100,Nscenarioo,Nattri,1))

#Contarst martix:C
contrast_val = -1/(Nalter - 1)
C = np.full((Nalter, Nalter), contrast_val)
np.fill_diagonal(C, 1) 
C=C.reshape(1,1,Nalter,Nalter)

##L matrix
L=np.zeros((Nido+100,Nscenarioo,Nalter-1,Nalter))
indmax=y_choice_1 #the chosen alternative
for i1 in range(0,Nido+100):
    for i2 in range(0,Nscenarioo):
        i4=0
        for i3 in range(0,Nalter):
            if i3==indmax[i1,i2,0]:
                L[i1,i2,:,indmax[i1,i2,0]]=np.ones(Nalter-1) #1s for chosen alternative (largest preference value)
            else:
                L[i1,i2,i4,i3]=-1
                i4+=1   #negative identity martix for others

###################fuction to calculate DFT############################
def calsqdistance(M):
    X=np.zeros((M.shape[0],M.shape[1],M.shape[2],M.shape[2]))
    for i in range(0,M.shape[2]-1):
        for j in range(i+1,M.shape[2]):
            X[:,:,i,j]=np.sum(((M[:,:,i,:]-M[:,:,j,:]))**2,axis=2)
            X[:,:,j,i]=np.sum(((M[:,:,j,:]-M[:,:,i,:]))**2,axis=2)
    return X

#define the prduct of feedback matrix 
def cals1(S,t):
    temp=np.zeros(([S.shape[0],S.shape[1],S.shape[2]]))
    for i in range(0,S.shape[0]):
        A=mat(S.shape[1],S.shape[2], fill.zeros)
        for k1 in range(0,S.shape[1]):
            for k2 in range(0,S.shape[2]):
                A[k1,k2]=S[i,k1,k2]
        eigval = mat()
        eigvec = mat()
        eig_sym(eigval, eigvec, A)
        SB=powmat(diagmat(eigval), t)
        OUT=eigvec*SB*eigvec.t()
        for k1 in range(0,S.shape[1]):
            for k2 in range(0,S.shape[2]):
                temp[i,k1,k2]=np.real(OUT[k1,k2])            
    return temp


def cal_psi(w_p):
    temp=np.zeros((w_p.shape[0],w_p.shape[1],w_p.shape[2],w_p.shape[2]))
    for i in range(0,w_p.shape[0]):
        for j in range(0,w_p.shape[1]):
            temp[i,j]=(np.diag(w_p[i,j].flatten())-(w_p[i,j]@w_p[i,j].T))
    return temp
  
def is_pos_def(x):
    return ~np.all(np.linalg.eigvals(x) > 0)

def check_p(mu1,ta1o,Nalter):
    if (np.sum(np.isnan(ta1o))!=0) or (np.sum(np.isnan(mu1))!=0) or (np.sum(~np.isfinite(ta1o))!=0) or (np.sum(~np.isfinite(mu1))!=0) or (np.sum(ta1o==np.Inf)!=0) or (np.sum(mu1==np.Inf)!=0):
        p=1e-50

    elif (np.sum(ta1o==0)==1) or (np.sum(np.diag(ta1o)<=0)!=0) or (np.sum(np.abs(mu1.flatten()))<0.0000000001) or (is_pos_def(ta1o)):
        p=1/Nalter
    else:
        p=multivariate_normal.cdf(x=mu1.flatten(),cov=ta1o)
    
    return p

def trans(params,cov):
    sample_p=np.random.multivariate_normal(params, cov, 1000000)
    phi1 = 1/(1+np.exp(-sample_p[:,0]));phi2=1/(1+np.exp(-sample_p[:,1]));t_step=1+np.exp(sample_p[:,5])
    return np.std(phi1),np.std(phi2),np.std(t_step)

###LLH
def cal_LL(param, C, data_att, w_p,Nid, Nscenario,Nalter,Nattri,L):
    phi1 = 1/(1+np.exp(-param[0]))
    phi2=1/(1+np.exp(-param[1]))
    use_scaling=1#True
    M=data_att
    
    if use_scaling:
        M=M*(np.array([param[2],param[3],param[3],param[4]]))
       
    else:
        w_p=(np.exp(np.append(param[2:5],0))/np.sum(np.exp(np.append(param[2:5],0)))).reshape(Nattri,1)
        
    
    p_0=np.array([[0],[param[6]]])
    Distsq=calsqdistance(M)
    epsilon=1
    #feedback matrix: S
    S=np.eye(Nalter)-phi2*np.exp(-phi1*Distsq) 
    Sprob_E=np.zeros((Nid,Nscenario)) #choice probability

    t_step=1+np.exp(param[5]) 
    #check coeff
    if t_step<1:
        t_step=1
    if phi1 < 1e-07:
        phi1 = 1e-07
    if np.abs(phi2) < 1e-07:
        phi2 = 0
    
    if np.abs(phi2) >= 0.999: 
        P = -np.log(1e-50)*Nid*Nscenario 
        return P
    # calculate Expectation
    else:
        Scheck2 = (1-phi2) * Nalter+ 0.0000000001
        Scheck=np.sum(np.sum(np.abs(S),axis=3),axis=2) 
        Scheck3=np.sum(np.sum(np.abs(Distsq),axis=3),axis=2) 
        
        use_real=1 #1： for each person
        if use_real:
            mu=w_p#.reshape((1,1,Nattri,1)) #expectation of the weight matrix
            psi=cal_psi(w_p)
        else:
            mu=w_p.reshape((1,1,Nattri,1)) #expectation of the weight matrix
            psi=(np.diag(w_p.flatten())-(w_p@w_p.T))
        EP=np.zeros((M.shape[0],M.shape[1],M.shape[2],1))
        COVP=np.zeros((M.shape[0],M.shape[1],M.shape[2],M.shape[2]))
        
        twmp1=Scheck<Scheck2
        if np.sum(twmp1)>=1:
            EP[twmp1]=(t_step*((C@M@mu)+ p_0))[twmp1]
            COVP[twmp1]=(t_step*(C@M@psi@M.transpose(0,1,3,2)@C.transpose(0,1,3,2)+(epsilon**2)*np.eye(Nalter)))[twmp1]
        
        twmp2=Scheck3<0.000001
        if np.sum(twmp2)>=1:
            EP[twmp2]=(t_step*((C@M@mu)+p_0))[twmp2]
            COVP[twmp2]=(t_step*(C@M@psi@M.transpose(0,1,3,2)@C.transpose(0,1,3,2)+(epsilon**2)*np.eye(Nalter)))[twmp2]
        
        
        twmp3= ~((Scheck<Scheck2)+(Scheck3<0.000001))
        if np.sum(twmp3)>=1:
            S=S[twmp3];M=M[twmp3]
            if use_real:
                mu=mu[twmp3];psi=psi[twmp3]
            NS=cals1(S,t_step)
            EP[twmp3]=np.linalg.inv(np.eye(Nalter)-S)@(np.eye(Nalter)-NS)@(C@M@mu)+ NS@p_0
        
            Z=np.zeros((S.shape[0],Nalter*Nalter,Nalter*Nalter))
            Fai=C@M@psi@M.transpose(0,2,1)@C.transpose(0,1,3,2)+(epsilon**2)*np.eye(Nalter)
            FaiN=Fai.reshape((S.shape[0],Nalter*Nalter,1))
            for iid in range(0,S.shape[0]):
                Z[iid,:,:]=np.kron(S[iid,:,:],S[iid,:,:])
            NZ=cals1(Z,t_step) 
            COVP[twmp3]=(np.linalg.inv(np.eye(Nalter*Nalter)-Z)@(np.eye(Nalter*Nalter)-NZ)@FaiN).reshape(S.shape[0],Nalter,Nalter)

        ##probility
        TAO=L@EP #expectaion
        sanjiao=L@COVP@L.transpose(0,1,3,2) #cov
        
        #multivariate normal distribution   
        for iid in range(0,Nid):
            for ic in range(0,Nscenario): 
                Sprob_E[iid,ic]=check_p(TAO[iid,ic,:],sanjiao[iid,ic,:,:],Nalter)
        Sprob_E[Sprob_E<=0.0001]=1e-50
        LLH=-np.sum(np.log(Sprob_E)) #log likelihood function  
        
        return LLH
 
    
#################################estimation####################################
Nfeval = 1
bounds=((-3,3),(-np.Inf,np.Inf),(0.1,20),(0.1,20),(0.1,20),(-6,np.Inf),(-np.Inf,np.Inf))

def callbackF(Xi):
    global Nfeval
    print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}    {7: 3.6f}     {8: 3.6f}  '.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3],Xi[4], Xi[5], Xi[6], cal_LL(Xi,C, data_att, w_p,Nido+100, Nscenarioo,Nalter,Nattri,L)))
    Nfeval += 1
 
print  ('{0}      {1}       {2}        {3}        {4}        {5}        {6}        {7}       {8}'.format('Iter', ' phi1', ' phi2',  ' beta_rc', ' beta_oc', ' beta_dr', 't_step','ASC_EV','-llh'))     

initial=np.array([-0.756112459,0.012100603,2.485598908,2.396882971,2.147511556,-2.501633752,0.880210399])
resOpt = sp.optimize.minimize(
        fun = cal_LL,
        x0 = initial,
        args = (C, data_att, w_p,Nido+100, Nscenarioo,Nalter,Nattri,L),
        method ='L-BFGS-B' ,       
        callback=callbackF,
        tol=0.001,
        bounds=bounds,
        options = {'disp': True}
        )

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

np.random.seed(28)
EST=resOpt['x'].copy()#Estimtaed parameter
EST[0] = 1/(1+np.exp(-resOpt['x'][0]));EST[1] = 1/(1+np.exp(-resOpt['x'][1]));EST[5] = 1+np.exp(resOpt['x'][5])
ERR=(np.sqrt(np.diag(resOpt.hess_inv.todense()))).copy() #Standard erro
ERR[[0,1,5]]=trans(resOpt['x'],resOpt.hess_inv.todense())
Z_VAL=EST/ERR #z-value
P_VAL=2*norm.cdf(-np.abs(Z_VAL)) #p-value
SIG=trans_significance(P_VAL) #singificant level


###########################################return results########################################
info = f"""
Estimation summary for DFT without eye-tracking
------------------------------------------------------------------------------------
Coefficient           Estimate      Std.Err.       z-val         P>|z|
------------------------------------------------------------------------------------
intercept.EV      {EST[6]:10.2f}   {ERR[6]:10.2f}     {Z_VAL[6]:10.2f}     {P_VAL[6]:10.2f} {SIG[6]}
\u03B2_RC              {EST[2]:10.2f}   {ERR[2]:10.2f}     {Z_VAL[2]:10.2f}     {P_VAL[2]:10.2f} {SIG[2]}
\u03B2_OC              {EST[3]:10.2f}   {ERR[3]:10.2f}     {Z_VAL[3]:10.2f}     {P_VAL[3]:10.2f} {SIG[3]}
\u03B2_DR              {EST[4]:10.2f}   {ERR[4]:10.2f}     {Z_VAL[4]:10.2f}     {P_VAL[4]:10.2f} {SIG[4]}
\u03C6_1               {EST[0]:10.2f}   {ERR[0]:10.2f}     {Z_VAL[0]:10.2f}     {P_VAL[0]:10.2f} {SIG[0]}
\u03C6_2               {EST[1]:10.2f}   {ERR[1]:10.2f}     {Z_VAL[1]:10.2f}     {P_VAL[1]:10.2f} {SIG[1]}
t                 {EST[5]:10.2f}   {ERR[5]:10.2f}     {Z_VAL[5]:10.2f}     {P_VAL[5]:10.2f} {SIG[5]}
------------------------------------------------------------------------------------
Significance:  0.01 '***' 0.1 '**' 0.317 '*' 

Log-Likelihood={-resOpt['fun']:10.2f}
BIC={len(initial)*np.log(Nid*Nscenario+Nido*Nscenarioo)+2*resOpt['fun']:10.2f}

"""

# Print the formatted information
print(info)

