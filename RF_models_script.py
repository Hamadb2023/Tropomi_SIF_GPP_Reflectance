#!/usr/bin/env python
# coding: utf-8

# In[1]:


#####################################################################################
#########                                                              #############
#########   Establishing the four Random Forest Model for understang   ##############
########## the synergy between SIF, reflectance and vegetation index   #############
#########                                                              #############
####################################################################################


# In[2]:



# Import required libraries

import warnings
warnings.filterwarnings('ignore')

import glob
from datetime import datetime, timedelta 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import linregress
import scipy
import math
from math import sqrt

from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import r2_score

import statsmodels.formula.api as smf
import statsmodels as sm

#get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[3]:


# Controling the label, font, axes and legend sizes
#plt.rc('font', size=16, weight='bold') #controls default text sizesns.set_style('whitegrid')
plt.rc('font', size=16) #controls default text sizesns.set_style('whitegrid')
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
plt.rc('legend', fontsize=20) #fontsize of the legend
plt.rc('axes', titlesize=20) #fontsize of the title
plt.rc('axes', labelsize=20) #fontsize of the x and y labels


# In[4]:


# Set up your working directory
import os
os.getcwd()
os.chdir("D:/SIF_GPP_PRI_Tropomi/Linear_Regression_output/LR_Stats/LAST_testing/figures")


# In[5]:


# load your merged data set

df_merge = pd.read_csv('D:\LCC/DataGPPfiltered.csv')

# convert your datetime into datetime format and set it as index 
df_merge['Timestamp'] = pd.to_datetime(df_merge['Timestamp'], format ='%m/%d/%Y')

#Filtering based on the distance (<=5 km) and cloud fraction (<=15%)
df_merge = df_merge[(df_merge['distance']<= 5)&(df_merge['cloud_fraction']<=0.15)]

# Outliers filtering
df_merge = df_merge[(df_merge['Err_mesure']<=0.15)&(df_merge['Err_mesure']>=-0.15)]

# Drop your columns 
df_merge['Timestamp'] = df_merge.set_index(df_merge['Timestamp'], inplace = True)
df_merge = df_merge.drop(columns = ['Timestamp'])

# Convert categorical variables as numeric, Biomes and Sites.
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# label_encoder object
label_encoder =LabelEncoder()
# Encode labels in column. 
df_merge['IGBP_site']= label_encoder.fit_transform(df_merge['Biome_site'])
df_merge['Site']= label_encoder.fit_transform(df_merge['Site_name'])

#Rename daily averaged SIF
df_merge.rename(columns={'daily_averaged_SIF':'SIF_Daily'}, inplace = True)

# convert site name to liste and sort the data frame
listSites = df_merge["Site_name"].unique().tolist()
listSites = sorted(listSites)

# replace null GPP values as nan
df_merge['GPP'][df_merge['GPP']==0] = np.nan

#Drop nan values 
df_merge.dropna(subset = ['GPP'], inplace = True)

#get dayofyear from dataframe index
df_merge['DOY'] = df_merge.index.dayofyear

df_merge


# In[8]:


# generate a new data set based on the correlation matrix results to establish the RF models

columns = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B11','B13',
           'SIF_Daily', 'GPP', 'NDVI', 'NIRv','PRI13','IGBP_site','Year','day','DoY','Site_name',
           'Site_palette','Biome_site']

Data = df_merge[columns]
#Data = FR_Fon[columns]
Data= Data.dropna(axis =0)
#Data = Data[(Data['IGBP_LC5km']=='CRO')]
Data


# In[11]:


# preparation for the random forest model setting
#get the required libraries 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

#select your explanatory variables

columns = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B11','B13',
           'SIF_Daily', 'NDVI', 'NIRv','PRI13','IGBP_site','Year','day','DoY','Site_name','Biome_site']

# select your target variable
columntarg = ['GPP']
data = Data[columns]
#data['B15'] =data['B15'].astype(float)
target = Data[columntarg]
# split your dataset into training set and testing set: 80% for training and 20% for testing the model 
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size= 0.20, random_state= 42)

#################################################################################################################################
#create a new data frame for testing data as you will need it to test your model not only on all data pooled across all sites but also based on each site and each PFT
df = pd.concat([data_test, target_test], axis =1)

#df.to_csv('data_testing2.csv') #uncomment this line to save your testing data in your working directory.


# In[12]:



#%%time

from sklearn.model_selection import train_test_split
#############################################################################################################################
#### Establishing the four RF models    ###################################################################################
##########################################################################################################################
columns = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B11','B13'] # uncomment this line to run your RF-R model
#columns = ['SIF_Daily','B1','B2','B3','B4','B5','B6','B7','B8','B9','B11','B13'] # uncomment this line to run your RF-SIF-R model
#columns = ['SIF_Daily','B1','B2','B3','B4','B5','B6','B7','B8','B9','B11','B13', 'IGBP_site']
#columns = ['SIF_Daily','NIRv','NDVI','PRI13']  # # uncomment this line to run your RF-SIF-VI model

######################################################################################################################
columntarg = ['GPP']
data = Data[columns]
target = Data[columntarg]

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size= 0.20, random_state= 42)


## Importing Random Forest regressor from the sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

## Use Randomized SearchCV to tune your model parameters

from sklearn.model_selection import RandomizedSearchCV

Randomized_GridCV = {'bootstrap': [True, False],# method used to sample data points
            'max_depth': [5, 6, 7, 8, 9,10, 15, 20, 30, 40, 50, 60,
                        70, 80, 90, 100, 110,120], # maximum number of levels allowed in each decision tree
            'max_features': ['log2', 'sqrt','auto'], # number of features in consideration at every split
            'min_samples_leaf': [1, 2, 3, 4, 5, 10], # minimum sample number that can be stored in a leaf node
            'min_samples_split': [1, 2, 6, 10], # minimum sample number to split a node
            'n_estimators': [2, 5, 10, 15, 20, 20, 30, 50, 80, 100, 150, 200]}, # number of trees in the random forest

rf_random = RandomizedSearchCV(cv=10, estimator=RandomForestRegressor(), n_iter=20,
                   n_jobs=-1,
                   param_distributions= Randomized_GridCV,
                   random_state= 42, verbose=2)
# Fit your random forest model

rf_random.fit(data_train, target_train)

# print the best parameters
print ('Best Parameters: ', rf_random.best_params_, ' \n')

# use the best parameters for prediction

best_model = rf_random.best_estimator_

GPP_RF = best_model.predict(data_test)
print("RF_model_score=%.6f"% best_model.score(data_train, target_train))


# In[23]:



#%%time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
sns.set_style('whitegrid')

#############################################################################################################################
#### Establishing the four RF models    ###################################################################################
##########################################################################################################################
columns = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B11','B13'] # uncomment this line to run your RF-R model
#columns = ['SIF_Daily','B1','B2','B3','B4','B5','B6','B7','B8','B9','B11','B13'] # uncomment this line to run your RF-SIF-R model
#columns = ['SIF_Daily','B1','B2','B3','B4','B5','B6','B7','B8','B9','B11','B13', 'IGBP_site']
#columns = ['SIF_Daily','NIRv','NDVI','PRI13']  # # uncomment this line to run your RF-SIF-VI model

######################################################################################################################

columntarg = ['GPP']
data = Data[columns]
#data['B15'] =data['B15'].astype(float)
target = Data[columntarg]

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size= 0.20, random_state= 42)

## Importing Random Forest Classifier from the sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

from sklearn.model_selection import RandomizedSearchCV

# use the best parameters to make model prediction
best_model = RandomForestRegressor(n_estimators = 200, min_samples_split = 10, min_samples_leaf= 1, max_features = 'sqrt', max_depth= 40, bootstrap= False) 
#best_model = RandomForestRegressor(n_estimators = 200, min_samples_split = 2, min_samples_leaf= 1, max_features = 'sqrt', max_depth= 110, bootstrap=False) 


best_model.fit(data_train, target_train)


GPP_RF = best_model.predict(data_test)

print("RF_Prediction_score=%.6f"% best_model.score(data_test, target_test))
print("RF_model_score=%.6f"% best_model.score(data_train, target_train))
#print("params:", regr_rf.estimator_params)
#print("alpha:", regr_rf.ccp_alpha)

# Model Evaluation Parameters

MSE = mean_squared_error(target_test,GPP_RF)
RMSE = math.sqrt(MSE)
MAE = mean_absolute_error(target_test, GPP_RF)
EVS = explained_variance_score(target_test, GPP_RF, multioutput='variance_weighted')
R_squared = best_model.score(data_test, target_test)

# Calculate the adjusted R-squared
adjusted_R = 1 - (1-best_model.score(data_test, target_test))*(len(target_test)-1)/(len(target_test)-data_test.shape[1]-1)

print("Adjusted R_squared:", adjusted_R)
print("RMSE:",RMSE)
print("MAE:", MAE)
print("Explained_variance_score:",EVS )

target_test['GPP_RF_R'] = best_model.predict(data_test)

#target_test.to_csv('GPP_Predicted_RF_R.csv') # uncomment this line to record your model GPP prediction in your working directory.


# In[24]:


#%%time

#######################################################################################################
########### Determining the feature relative importance        #######################################
######################################################################################################
# Get your feature_importances and mean permutation importances from the best model RF fit

#importances = best_model.feature_importances_
importances = best_model.feature_importances_

indices = np.argsort(importances)
features = data_train.columns
imp = []

dat = pd.DataFrame()
imp.append(importances)
########################################################################################################################

Dat= pd.DataFrame(imp).transpose()

#Dat = pd.concat([dfm, dfmean, std], axis =1)
Dat.columns = ['Importances']
Dat.index = data_train.columns
Dat.index.name ='Variables'

#rename columns 
#Dat.rename(columns ={'PRI13':'PRI', 'IGBP_LCnum':'IGBP_Biome'})
Dat.rename(index={'PRI13':'PRI','IGBP_site':'IGBP_Biome','SIF_Daily':'SIF Daily', 'GPP':'GPP'}, inplace = True)
Dat.reset_index(inplace = True)

Dat.sort_values(by= ['Importances',], inplace= True )

#print(Dat.index.name)
#Dat.to_csv('RF_R_Importances.csv') # uncomment this line to record your feature relative importance into your working directory
Dat


# In[ ]:




