# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 17:43:32 2022
SSAI Interview Data Comparison 
@author: Mitch Creelman
"""

#%% Load packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from string import ascii_letters

#%% Adjustables

# Set File Path
path = "C:/Users/Owner/Documents/School/Spring 2022/Interview Questions/SSAI_Task_Data/"

# use subsample( 1 = yes, 0 = no)
# use this to use a smaller subset until we are ready to do the whole thing
sub = 0


# turn on seaborn
sns.set()

#%% Read in the csv's

# data
test = pd.read_csv(path + "testing.csv")  # testing Data
train = pd.read_csv(path + "training.csv") # training data
val = pd.read_csv(path + "validation.csv") # validation Data

if sub == 1:
    train = train.head(300)



# individual indexing without label
L = list(val.columns[:-1]) # wavelengths of samples
Chl = val.columns[-1:]

out = 'C:/Users/Owner/Documents/School/Spring 2022/Interview Questions/SSAI_Task_Data/Output/Figures/'



#%% create Graph of each band vs. Chl
d = val # set the data
 
# Normalize the data 
column_maxes = d[L].max() # find the max of each column
# d_max = column_maxes.max() 
normalized_d = d[L] / column_maxes # express the dependent variables as a %

d[L] = normalized_d
#%%
fig, axes = plt.subplots(nrows=4, 
                         ncols=4,
                         sharex = True, 
                         sharey = True,
                         figsize=(14,8))

i = 0
n = 0 

for row in axes:
    if i == 0:
        p = 0
    else:
        p = p + 1
        
    i = 0
    for col in row:
        # col.plot(x[L[i]], y)
        f = sns.regplot(x = d[Chl] ,
                         y = L[n],
                         data = d,
                         scatter_kws={"s":1},
                         ax = axes[p,i])
        
        # Calculate and print r^2 and p values
        x = d[Chl].to_numpy().squeeze()
        y = d[L[n]].to_numpy().squeeze()
        r, k = sp.stats.pearsonr(x, y)
        ax = axes[p,i]
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, k),
        transform=ax.transAxes)
        
        # hide the labels and axis
        f.set(xlabel=None, ylabel = None)
        
        # set the graph name to the individual 
        col.set_title(str(L[n]), pad = 0.5)
        print(str(p) + str ( i ))
        n = n + 1
        i = i + 1
        
plt.suptitle('Chlorophyll-a vs. Reflectance by Wavelength',size = 25)
fig.supylabel('Reflectance (% of maximum)', size = 20)
fig.supxlabel('Chlorophyll-a Levels', size = 20)


plt.show()
ig.savefig(out + 'Correlation_By_Band.png', dpi = 900)
#%% Correlation Matrix Plot
d = val


# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)

plt.suptitle('Correlation Matrix for Chlorophyll Concentrationv \n' + 
             'and Band Reflectance',
             size = 20 )

fig.savefig(out + 'Correlation_Matrix.png', dpi = 900)