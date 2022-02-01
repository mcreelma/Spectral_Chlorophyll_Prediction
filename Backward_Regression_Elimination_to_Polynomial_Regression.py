
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:15:59 2022

SSAI Interview Regression Selection

@author: Mitch Creelman
"""

#%% Load packages

import matplotlib.pyplot as plt
# import math
import numpy as np
import os
import pandas as pd
# import scipy as sp
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
# from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

#%% Adjustables

# Set File Path
path = "C:/Users/Owner/Documents/School/Spring 2022/Interview Questions/SSAI_Task_Data/"

# number of elements you want to use in the model
els = 3

# use subsample( 1 = yes, 0 = no)
# use this to use a smaller subset until we are ready to do the whole thing
sub = 0
samples = 15000 # set number of samples


# turn on seaborn
sns.set()

# degrees of fit
deg = 8
# ..... Because this one goes up to 11.......

# do you want to save the outputs to a csv?
save = 1

# plot the distribution of outputs
plot = 0
#%% Read in the csv's

# data
test = pd.read_csv(path + "testing.csv")  # testing Data
train = pd.read_csv(path + "training.csv") # training data
val = pd.read_csv(path + "validation.csv") # validation Data

# initialize subset
if sub == 1:
    train = train.sample(samples)

# clean outliers from training data
train = train[(np.abs(stats.zscore(train)) < 3).all(axis=1)]
train = train.dropna() 

#%% Define the ability to index the element


def find(s, el):
    for i in s.index:
        if s[i] == el: 
            return i
    return None


#%% Make Subfolder for Outputs


# file outputs
out = path + 'Output/'
try:
    os.makedirs(out)
except OSError:
    print ("Did not create %s " % out)
else:
    print ("Successfully created the directory %s" % out)

# figure outputs
figs_out = out + 'Figures/'

try:
    os.makedirs(figs_out)
except OSError:
    print ("Did not create %s " % figs_out)
else:
    print ("Successfully created the directory %s" % figs_out)
    
    
#%% Pull out headers


x_columns = list(val.columns[:-1]) # wavelengths of samples
y_columns = val.columns[-1:] # CHANGE THIS TO -2 IF THIS DOESN'T WORK

# set the data for the 
data = val
y = data[y_columns]

i = len(val.columns)-(els+1) # set indexing to leave us with 3

# Go through and reduce the whole thing down to 2 variables based on 
# eliminating the highest p values
for p in range (i):
    x = data[x_columns]
    results = sm.OLS(y, x).fit() # fit the model
    ANDMYAXE = find( results.pvalues , results.pvalues.max()) # find the variable 
    x_columns.remove(ANDMYAXE) # Remove it from the dataframe

    
#%% Regression and Graphing
TGroups = x_columns # transfer variable to old code variable

#%% Create Linear Regression Model

x = train[TGroups]
y = train[y_columns]


pre_process = PolynomialFeatures(degree = deg) # set the number of degrees for the polynomial
x_poly = pre_process.fit_transform(x) # transform the array above to a set of 2nd degree predictions

pr_model = LinearRegression() # 
pr_model.fit(x_poly, y) # 
y_pred = pr_model.predict(x_poly)

b = pr_model.intercept_ # intercept of data
m = pr_model.coef_[0] # array of all the variables
#%% Repeat this for the validation side side

x_poly_val = pre_process.fit_transform(val[TGroups]) # transform the array above to a set of 2nd degree predictions
y_pred_val = pr_model.predict(x_poly_val)

#%% Model the testing data

x_poly_test = pre_process.fit_transform(test[TGroups]) # transform the array above to a set of 2nd degree predictions
y_pred_test = pr_model.predict(x_poly_val)


#%% Add Chl levels and residuals for validation data

# Training Data
train['Calculated Chl'] = y_pred # create a new column with the modeled chlorophyl values
# train['Residuals'] = abs(train['Calculated Chl'] - train['Chl']) # get the residuals
train['Residuals'] = train['Calculated Chl'] - train['Chl'] # get the residuals
# Save the new data to a csv
if save == 1:
    train.to_csv(out + 'Training_Output.csv')


# Validation Data
val['Calculated Chl'] = y_pred_val # create a new column with the modeled chlorophyl values
# val['Residuals'] = abs(val['Calculated Chl'] - val['Chl']) # get the residuals
val['Residuals'] = val['Calculated Chl'] - val['Chl'] # get the residuals
# Save the new data to a csv
if save == 1:
    val.to_csv(out + 'Validation_Output.csv')


# Testing data
test['Chl'] = y_pred_test # create a new column with the modeled chlorophyl values
# Save the new data to a csv
if save == 1:
    test.to_csv(out + 'Final_Answer.csv')
    

#%% Testin the range out 

if sub == 1:
    print('The sample size is ' + str(samples))
else:
    print('The sample size is ' + str(len(train)))

print('The modeled power is ' + str(deg))
print('The max modeled value is ' + str(round(test['Chl'].max(),3)))
print('The min modeled value is ' + str(round(test['Chl'].min(),3)))
print(x_columns)
r = abs(test['Chl'].max()  - test['Chl'].min())
print('The range of modeled concentrations is ' + str(round(r,3)) )

#%% Find the RMSE

#Root Mean Squared Error
realVals = val['Chl']
predictedVals = val['Calculated Chl']
rmse = mean_squared_error(realVals, predictedVals, squared = False)
print('The val RMSE of power ' + str(deg) + ' is ' +str(round(rmse,3)))

#Mean Squared Error
realVals = val['Chl']
predictedVals = val['Calculated Chl']
mse = mean_squared_error(realVals, predictedVals)
print('The val MSE of power ' + str(deg) + ' is ' +str(round(mse,3)))
#%% RMSE training

#%% Find the RMSE

realVals = train['Chl']
predictedVals = train['Calculated Chl']
# mse = mean_squared_error(realVals, predictedVals)
# If you want the root mean squared error
rmse = mean_squared_error(realVals, predictedVals, squared = False)
print('The train RMSE of power ' + str(deg) + ' is ' +str(round(rmse,3)))

realVals = train['Chl']
predictedVals = train['Calculated Chl']
mse = mean_squared_error(realVals, predictedVals)
# If you want the root mean squared error
# rmse = mean_squared_error(realVals, predictedVals, squared = False)
print('The train MSE of power ' + str(deg) + ' is ' +str(round(mse,3)))