# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:09:10 2022

Comparative plots

@author: Mitchell Creelman
"""
#%% Load packages

import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import scipy as sp
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

#%% Adjustables

# Set File Path
path = "C:/Users/Owner/Documents/School/Spring 2022/Interview Questions/SSAI_Task_Data/Output/"

# turn on seaborn
sns.set()

# save the plots
save = 1

# use this to use a smaller subset until we are ready to do the whole thing
sub = 1

samples = 15000 # set number of samples

#%% Read in the csv's

# data
test = pd.read_csv(path + "Final_Answer.csv")  # testing Data
train = pd.read_csv(path + "Training_Output.csv") # training data
val = pd.read_csv(path + "Validation_Output.csv") # validation Data

# initialize subset
if sub == 1:
    train = train.sample(samples)


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
    


#%% create a big boy in order to be able to graph by group

# Set the order for the columns

order = ['# 400', '412.5', '442.5', '490', '510', '560', '620', '665', '673.75',
  '681.25', '708.75', '753.75', '761.25', '764.375', '767.5', '778.75', 
  'Group', 'Chl', 'Calculated Chl', 'Residuals']


# add column labeling which dataset it is

# Testing values
test['Group'] = pd.Series(["Testing" for x in range(len(test.index))]) # add new column
test['Calculated Chl'] = pd.Series([ np.nan for x in range(len(test.index))]) # add new column
test['Residuals'] = pd.Series([ np.nan for x in range(len(test.index))]) # add new column
test = test[order] # reorder

# training values
train['Group'] = pd.Series(["Training" for x in range(len(train.index))]) # add new column
train = train[order] # reorder

# validation values
val['Group'] = pd.Series(["Validation" for x in range(len(val.index))]) # add new column
val = val[order] # reorder


# concatenate them all into a single df
frames = [train, val] # list all the df's
comb = pd.concat(frames, # concatenate them into a single dataset
                 ignore_index=True) # ignore the larger index
comb = comb.squeeze() # Squeeze everything in

print(comb.columns)



#%%


# p = sns.lmplot(x="Chl", y="Residuals", hue="Group", data=comb);
# p.plot_marginals(sns.histplot, element="Group", color="#03012d")
# #%% Join Trend Plot 
my_pal = {"Training": "peru", "Validation": "g"}


p = sns.jointplot(data=val, x="Chl", y="Calculated Chl", 
                  # hue="Group", 
                   kind = 'reg', 
                  # marker="+",
                  # scjointplot={'s': 1},
                  # palette = my_pal,
                  color = 'g',
                  hue_order=(['Validation', 'Training']))

p.fig.subplots_adjust(top=0.9) # Reduce plot to make room 
p.fig.suptitle("Chlorophyll-a Concentrations \n" + 
               "For Validation Dataset", fontsize = 16)
# p.ax.fig.subplots_adjust(top=0.9) # Reduce plot to make room 
p.ax_joint.set_xlabel('Measured Concentrations', fontsize = 15)
p.ax_joint.set_ylabel('Modeled Concentrations', fontsize = 15)
# p.ax_joint.collections[0].set_alpha(0)

# p.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)

if save == 1:
    # set the name of the figure
    name_out = 'Validation_Mod_v_Measure.png'
    fig = p
    fig.savefig(figs_out + name_out, dpi = 900 )
    # plt.close()
    
    
#%%
my_pal = {"Training": "peru", "Validation": "g"}


p = sns.jointplot(data=val, x="Chl", y="Residuals", 
                  # hue="Group", 
                   kind = 'reg', 
                  # marker="+",
                  # scjointplot={'s': 1},
                  # palette = my_pal,
                  color = 'g')


p.fig.subplots_adjust(top=0.9) # Reduce plot to make room 
# p.ax_marg_y.set_ylim(-2, 40)
p.fig.suptitle("Residuals for Validation Dataset", fontsize = 19)
p.ax_joint.set_xlabel('Measured Concentrations', fontsize = 15)
p.ax_joint.set_ylabel('Residaul Values', fontsize = 15)
# p.ax_joint.collections[0].set_alpha(0)

# p.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)

if save == 1:
    # set the name of the figure
    name_out = 'Validation_Resid_v_Measure.png'
    fig = p
    fig.savefig(figs_out + name_out, dpi = 900 )
#%%


comb["all"] = ""
plt.figure()
my_pal = {"Training": "peru", "Validation": "g"}


ax = sns.violinplot(x="all", y="Residuals", 
                    hue="Group",
                    data=comb, 
                    palette = my_pal,
                    split=True,
                    legend_out = True)
ax.set_xlabel("")
ax.set_title('Distribution of Residuals for \n '
          + 'Multivariate 8th Degree Polynomial Regression ')
plt.show()

if save == 1:
    # set the name of the figure
    name_out = 'Residual_Leaf.png'
    fig = ax.get_figure()
    fig.savefig(figs_out + name_out, dpi = 900 )
    # plt.close(fig)
    
    
    #%%
    

x = val['Chl'].to_numpy().squeeze()
y = val['Calculated Chl'].to_numpy().squeeze()
r, k = sp.stats.pearsonr(x, y)
# print('The model has produced a correlation regression between the modeled chlorophyll-a' + 
#       ' levels and the measured chlorophyll-a levels with a p value of ' + str(k)  + ' a pearson ' + 
#       ' r of ' + str(r) + )


x =  val['Chl'].values.reshape(-1,1)
y = val['Calculated Chl'].values.reshape(-1,1)
regr = linear_model.LinearRegression()
regr.fit(x, y)
print(regr.coef_[0])
print(regr.intercept_)