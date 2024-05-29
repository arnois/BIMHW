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
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture as GM
from matplotlib import pyplot as plt
from seaborn import heatmap
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

def get_factor_return_by_T(data_r: pd.DataFrame, data_X: pd.DataFrame, 
                           weekdate: str = '2009Jan02') -> np.ndarray:
    """
    Parameters
    ----------
    data_r: DataFrame
        Cross -section returns data for each asset.
    data_X: DataFrame
        Factor exposures data.

    Returns
    -------
    Factor returns as coefficients of regressing factor exposures against 
    asset returns.

    """
    # Cross-section weekly returns for given date
    r = data_r[weekdate]
    # Factor exposures
    X = data_X.drop('issue_id', axis=1)
    # (y,X) merged data to get rid of NAs
    model_data = pd.concat([r,X], axis=1)
    y = model_data.dropna()[weekdate]
    X = model_data.dropna().drop(weekdate, axis=1)
    # Model Fit
    lr = LinearRegression(fit_intercept=False).fit(X,y)
    # Factor's returns
    return lr.coef_

def plot_heatM(M: pd.DataFrame = None, plt_size: tuple = (10,8), 
               txtIn: bool = False, vmin: int = -1, vmax: int = 1,
               plot_title: str = '') -> None:
    """
    Returns: Heatmap plot.
    """    
    # Corrmatrix
    plt.figure(figsize=plt_size)
    heatmap(
        M,        
        cmap='RdBu', 
        annot=txtIn, 
        vmin=vmin, vmax=vmax,
        fmt='.2f')
    plt.title(plot_title, size=20)
    plt.tight_layout();plt.show()
    return None

#%% DATA IMPORT
data_X = pd.read_excel(r'Data_for_Excercise.xls', sheet_name='X')
data_r = pd.read_excel(r'Data_for_Excercise.xls', sheet_name='r')

#%% DATA MGMT
# Cluster number lookout for returns
n_comp = np.arange(2, 9)
## Weekdates without any returns at all...
outcols = ['issue_id','2012Jan13','2012Jan20','2012Jan27','2012Dec14']
## Returns without empty weekdate-returns
X_gmm = data_r.drop(outcols, axis=1).dropna()
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
idx_naRows = data_r.index[data_r.drop(outcols, axis=1).isna().any(axis=1)]
ngen = len(idx_naRows)
gmm_genX = pd.DataFrame(gmm.sample(ngen)[0], index=idx_naRows)
idxNA = np.where(data_r.drop(outcols, axis=1).isna())
data_r_gmm = data_r.drop(outcols, axis=1).copy()
for i,j in list(zip(idxNA[0], idxNA[1])):
    data_r_gmm.iloc[i,j] = gmm_genX.loc[i,j]
data_r_gmm.insert(0,'issue_id',data_r['issue_id'])

# Reiteration to generate full missing dates of returns...
X_gmm2 = data_r_gmm.drop('issue_id', axis=1).T
gmm2 = GM(n_opt, random_state=13).fit(X_gmm2)
# Fully date missing returns generation
idx_naCols = data_r.T.index[data_r.isna().all()]
ngen2 = len(idx_naCols)
gmm2_genX = pd.DataFrame(gmm2.sample(ngen2)[0], index=idx_naCols)
data_r_gmm2 = data_r_gmm.copy()
idx_missingCols = np.where(data_r.columns.isin(gmm2_genX.index))[0]
for i in range(len(idx_missingCols)):
    tmpr = gmm2_genX.iloc[i,:]
    data_r_gmm2.insert(int(idx_missingCols[i]), tmpr.name, tmpr)


#%% SINGLE XSECTION  FACTOR RETURNS
f1 = get_factor_return_by_T(data_r,data_X)

#%% XSECTION FACTOR RETURNS DROPPING NAs
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
F.index = X.columns

#%% XSECTION FACTOR RETURNS FILLING NAs
# Partial filled...
r = data_r_gmm.drop('issue_id', axis=1)
X = data_X.drop('issue_id', axis=1)
lr = LinearRegression(fit_intercept=False)
lr.fit(X,r)
F_ = pd.DataFrame(lr.coef_.T, columns = r.columns, index = X.columns)

# Full filled...
r = data_r_gmm2.drop('issue_id', axis=1)
X = data_X.drop('issue_id', axis=1)
lr = LinearRegression(fit_intercept=False)
lr.fit(X,r)
F__ = pd.DataFrame(lr.coef_.T, columns = r.columns, index = X.columns)

#%% FACTOR RETURNS CORRELATION AMONGS METHODS: Dropping vs partially filling NA
factor_rho = pd.DataFrame(index = F.index, columns = ['rho'])
for k in range(F.shape[0]):
    factor_rho.loc[factor_rho.index[k], 'rho'] = np.corrcoef(F.iloc[k,:], 
                                                             F_.iloc[k,:])[0,1]
# Viz missing values impact over F matrix
plt.figure(figsize=(12,8))
factor_rho.plot.bar(legend=None,xlabel='Factor',ylabel='Rho')
plt.tick_params(axis='x', labelrotation=90)
plt.xticks(fontsize=5)
plt.tight_layout();plt.show()

# Factors where missing data methods produce different returns
factor_rho[abs(factor_rho.rho) < 0.70].sort_values('rho', ascending=False)

#%% FACTOR RETURNS CORRELATION AMONGS METHODS: Dropping vs fully filling NA
factor_rho2 = pd.DataFrame(index = F.index, columns = ['rho'])
for k in range(F.shape[0]):
    factor_rho2.loc[factor_rho2.index[k], 'rho'] = np.corrcoef(F.iloc[k,:], 
                                                             F__[F.columns].\
                                                                 iloc[k,:])[0,1]

# Viz missing values impact over F matrix
plt.figure(figsize=(12,8))
factor_rho2.plot.bar(legend=None,xlabel='Factor',ylabel='Rho')
plt.tick_params(axis='x', labelrotation=90)
plt.xticks(fontsize=5)
plt.tight_layout();plt.show()
# Factors where fully missing data methods produce different returns
factor_rho2[abs(factor_rho2.rho) < 0.70].sort_values('rho', ascending=False)

#%% FACTOR COVARIANCE MATRIX

# Dropping NAs
rho_F = F.T.corr()
S_F = F.T.cov()
# partially filled NAs
rho_F_ = F_.T.corr()
S_F_ = F_.T.cov()
# fully filled NAs
rho_F__ = F__.T.corr()
S_F__ = F__.T.cov()
# Factor corr matrix differences
tmpD1 = rho_F - rho_F_
tmpD2 = rho_F - rho_F__
tmpD3 = rho_F_ - rho_F__
plot_heatM(tmpD1, vmin=tmpD1.min().min(), 
           vmax=tmpD1.max().max(), 
           plot_title='Dropping NAs vs Partially filled')
plot_heatM(tmpD2, vmin=tmpD2.min().min(), 
           vmax=tmpD2.max().max(), 
           plot_title='Dropping NAs vs Fully filled')
plot_heatM(tmpD3, vmin=tmpD3.min().min(), 
           vmax=tmpD3.max().max(), 
           plot_title='Partially filled vs Fully filled')

tmp = S_F__ - S_F_
plot_heatM(tmp, vmin=tmp.min().min(), 
           vmax=tmp.max().max(), 
           plot_title='Cov\nPartially filled vs Fully filled')

#%% RETURNS COVARIANCE
# returns
returns__ = data_r_gmm2.drop('issue_id', axis=1)
# Full
returns__.T.cov()
# XS_FX
X_F_X = X@S_F__@X.T
