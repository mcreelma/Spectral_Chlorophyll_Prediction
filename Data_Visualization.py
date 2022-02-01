# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:51:09 2022

SSAI Interview Question
 -- Data Visualization

@author: Mitch Creelman
"""


#%% Load packages

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

#%% Adjustables

# Set File Path
path = "C:/Users/Owner/Documents/School/Spring 2022/Interview Questions/SSAI_Task_Data/Output/"

# turn on seaborn
sns.set()

# create a proper order for the columns
order = ['# 400', '412.5', '442.5', '490', '510', '560', '620', '665', '673.75',
  '681.25', '708.75', '753.75', '761.25', '764.375', '767.5', '778.75', 
  'Group', 'Chl']
#%% Read in the csv's

# data
test = pd.read_csv(path + "Final_Answer.csv")  # testing Data
train = pd.read_csv(path + "Training_Output.csv") # training data
val = pd.read_csv(path + "Validation_Output.csv") # validation Data
#%% ouput paths 

# file outputs
out = path + 'Output/'
try:
    os.makedirs(out)
except OSError:
    print (" ")
else:
    print ("Successfully created the directory %s" % out)

# figure outputs
figs_out = out + 'Figures/'

try:
    os.makedirs(figs_out)
except OSError:
    print (" ")
else:
    print ("Successfully created the directory %s" % figs_out)
#%% create a big boy in order to be able to graph by group

# add column labeling which dataset it is
# Testing values
test['Group'] = pd.Series(["Testing" for x in range(len(test.index))]) # add new column
# test['Chl'] = pd.Series([ np.nan for x in range(len(test.index))]) # add new column
test = test[order] # reorder
# print Check
print('Testing column check')
print(val.head())
print( )
print( )

# training values
train['Group'] = pd.Series(["Training" for x in range(len(train.index))]) # add new column
train = train[order] # reorder
# print Check
print('Training column check')
print(train.head())
print( )
print( )

# validation values
val['Group'] = pd.Series(["Validation" for x in range(len(val.index))]) # add new column
val = val[order] # reorder
# print Check
print('Validation column check')
print(val.head())
print( )
print( )


# concatenate them all into a single df
frames = [test, train, val]
comb = pd.concat(frames)
comb = comb.squeeze()

# print(comb)

#%% Pull out headers

L = list(val.columns[:-1]) # wavelengths of samples
Chl = val.columns[-2:] # chlorophyl levels


#%% Wavelengths notched box
df = comb[L]
plt.figure()

df_long = pd.melt(df, "Group" , var_name="Wavelength (nm)", value_name="Reflectance Value")
ax = sns.factorplot("Wavelength (nm)", hue="Group", y="Reflectance Value", data=df_long, kind="box",
                    legend_out = False, aspect = 24/8, notch = True)

# axis and labels
ax.set_xticklabels(rotation = 0, fontsize = 15)
ax.set_yticklabels(fontsize = 15)
ax.set_xlabels(fontsize = 20)
ax.set_ylabels(fontsize = 20)
ax.fig.suptitle('Range of Cleaned Reflectance Values By Wavelength \n', fontsize = 25)
ax.fig.subplots_adjust(top=0.9) # Reduce plot to make room 
plt.show() 

# set the name of the figure
name_out = 'Bandwidth_Box_Plot.png'
fig = ax
fig.savefig(figs_out + name_out, dpi = 900)
# plt.close(fig)

#%% Wavelengths as a violin
df = comb[L]
plt.figure()

ax = sns.factorplot("Wavelength (nm)", hue="Group", y="Reflectance Value",
                    data=df_long, kind="violin",
                    legend_out = False, aspect = 24/8)
# axis and labels 
ax.set_xticklabels(rotation = 0, fontsize = 15)
ax.set_yticklabels(fontsize = 15)
ax.set_xlabels(fontsize = 20)
ax.set_ylabels(fontsize = 20)
ax.fig.suptitle('Range of Cleaned Reflectance Values By Wavelength \n', fontsize = 25)
ax.fig.subplots_adjust(top=0.9) # Reduce plot to make room 
plt.show()

# set the name of the figure
name_out = 'Bandwidth_Leaf_Plot.png'
fig = ax
fig.savefig(figs_out + name_out, dpi = 900)
# plt.close(fig)


#%% Chlorophyl Box

df = comb[Chl]
plt.figure()

df_long = pd.melt(df, "Group" , var_name="Wavelength", value_name="Value")
ax = sns.factorplot("Wavelength", hue="Group", y="Value", data=df_long, kind="box",
                    legend_out = True, notch = True )
# axis and labels
ax.set_xticklabels(fontsize = 0)
# ax.set_xlabels('Chlorophyll-a Levels',fontsize = 20)
ax.set_xlabels(' ')
ax.set_ylabels(fontsize = 20)
ax.fig.suptitle('Chlorophyll-a Levels by Group', fontsize = 25)
ax.fig.subplots_adjust(top=0.9) # Reduce plot to make room 
plt.show()

# set the name of the figure
name_out = 'Concentration_Box_Plot.png'
fig = ax
fig.savefig(figs_out + name_out, dpi = 900)
# plt.close(fig)

#%% Chlorophyl violin

df = comb[Chl]

plt.figure(dpi = 1800)

df_long = pd.melt(df, "Group" , var_name="Wavelength", value_name="Value")
ax = sns.factorplot(x = "Wavelength", hue="Group", y="Value", data=df_long, kind="violin",
                    legend_out = False)

#

ax.set_xticklabels(fontsize = 0)
# ax.set_xlabels('Chlorophyll-a Levels',fontsize = 20)
ax.set_xlabels(' ')
ax.set_ylabels('Concentration',fontsize = 20)
ax.fig.suptitle('Chlorophyll-a Levels by Group', fontsize = 25)
ax.fig.subplots_adjust(top=0.9) # Reduce plot to make room 
plt.show()

# set the name of the figure
name_out = 'Concentration_Leaf_Plot.png'
fig = ax
fig.savefig(figs_out + name_out, dpi = 900)
# plt.close(fig)
