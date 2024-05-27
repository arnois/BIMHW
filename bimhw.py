# -*- coding: utf-8 -*-
"""
Code for a basic factor model excercise

@author: arnulf.q@gmail.com
"""
#%% MODULES
import os
import sys
import pandas as pd
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

#%% SINGLE ID VIZ
tmpID = 59863
r = data_r['2009Jan02']
X = data_X.drop('issue_id', axis=1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression(fit_intercept=False)

lr.fit(X,r)
