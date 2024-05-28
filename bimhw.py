# -*- coding: utf-8 -*-
"""
Code for a basic factor model excercise

@author: arnulf.q@gmail.com
"""
#%% MODULES
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.mixture import GaussianMixture as GM
from matplotlib import pyplot as plt
#%% WORKING DIRECTORY
str_cwd = r'C:\Users\jquintero\Documents\GitHub\BIMHW' 
if not os.path.exists(str_cwd):
    # Current working path
    cwdp = os.getcwd()
    cwdp_parent = os.path.dirname(cwdp)
    sys.path.append(cwdp)
else:
    sys.path.append(str_cwd)
    
#%% UDF
def corrmatrix(df: pd.DataFrame = None, 
               corrM: str = 'spearman') -> pd.DataFrame:
    """
    Returns: Correlation matrix.
    """        
    # Corrmatrix
    E = df.corr(method=corrM.lower())

    return E

#%% DATA IMPORT
data_X = pd.read_excel(r'Data_for_Excercise.xls', sheet_name='X')
data_r = pd.read_excel(r'Data_for_Excercise.xls', sheet_name='r')

#%% DATA MGMT

# Returns 
# Cluster number lookout
n_comp = np.arange(2, 21)
outcols = ['issue_id','2012Jan13','2012Jan20','2012Jan27','2012Dec14']
X_gmm = data_r.drop(outcols, axis=1).dropna().T
lst_gmm = [GM(n, random_state=13).fit(X_gmm) for n in n_comp]
# GMM Models Comparison
gmm_model_comparisons = pd.DataFrame({
    "BIC" : [m.bic(X_gmm) for m in lst_gmm],
    "AIC" : [m.aic(X_gmm) for m in lst_gmm]},
    index=n_comp)

# Optimal components
n_opt = gmm_model_comparisons.\
    index[gmm_model_comparisons.apply(np.argmin)['BIC']]
## plotting...
plt.figure(figsize=(8,6))
ax = gmm_model_comparisons[["BIC","AIC"]].\
    plot(color=['darkcyan', 'b'], linestyle=':', marker='o', 
         mfc='w', xticks=gmm_model_comparisons.index)
plt.axvline(x=n_opt, color='orange', alpha=0.25)
plt.xlabel("Number of Clusters")
plt.ylabel("Score"); plt.show()

# Gen model for missing returns with joint dynamics
gmm = GM(n_opt, random_state=13).fit(X_gmm)
# Missing returns data generation
ngen = data_r.drop(outcols, axis=1).T.isna().sum().max()
gmm_genX = pd.DataFrame(ngen,columns=X_gmm.columns,index=X_gmm.index)
idxNA = np.where(data_r.drop(outcols, axis=1).T.isna())
data_r_gmm = data_r.T.copy()
data_r_gmm.iloc[np.unique(idxNA[0]),:]
#%% SINGLE ID VIZ
# Data seg
tmpID = '2009Jan02'
r = data_r[tmpID]
X = data_X.drop('issue_id', axis=1)
model_data = pd.concat([r,X], axis=1)
y = model_data.dropna()[tmpID]
X = model_data.dropna().drop(tmpID, axis=1)
# Model Fit
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=False)
lr.fit(X,y)
lr.coef_

#%% XSECTION FACTOR RETURNS
F = pd.DataFrame()
for tmpid in data_r.columns.drop('issue_id'):
    lr = LinearRegression(fit_intercept=False)
    r = data_r[tmpid]
    X = data_X.drop('issue_id', axis=1)
    model_data = pd.concat([r,X], axis=1)
    y = model_data.dropna()[tmpid]
    X = model_data.dropna().drop(tmpid, axis=1)
    if y.empty or X.empty:
        print(f'{tmpid} Empty (X,y)')
        continue
    lr.fit(X,y)
    F = pd.concat([F, pd.Series(lr.coef_).rename(tmpid)],axis=1)

    