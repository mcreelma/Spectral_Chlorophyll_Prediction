# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:49:37 2022

@author: Owner
"""

#%% Load packages

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid")
#%%
path = "C:/Users/Owner/Documents/School/Spring 2022/Interview Questions/SSAI_Task_Data/"
# file outputs
out = path + 'Output/'
# figure outputs
figs_out = out + 'Figures/'
#%% Range


x =  [1,2,3,4,5,6,7,8]
y = [64.08134540986411,
     64.31340651239526,
     59.77815714859389,
     58.29173448428628,
     58.05284336069599,
     64.88410425186157,
     74.203125,
     55.59375]

plt.plot(x,y) 

plt.yticks(fontsize= 9)
plt.title("Range of Modeled Values by Degree of Fit", fontsize = 15) 
plt.xlabel("Degree of Fit", fontsize = 12)
plt.xticks(fontsize = 9) 
plt.ylabel("Range of Modeled Concentrations", fontsize = 12) 


plt.show()




#%% RMSE 


fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Error of Fit by Degree')

x =  [1,2,3,4,5,6,7,8, 9 , 10, 11, 12]
yval = [8.473, 6.627, 6.744, 7.095, 7.476, 6.795, 6.872, 6.002, 5.997, 6.001, 
        6.001, 6.001
     ]
ytrn = [10.6, 9.064,  7.165, 5.706, 5.088, 4.799, 4.768, 4.52, 4.522, 4.522, 
        4.522, 4.522, 
        ]

ax1.plot(x, yval, label = "validation", color = 'g')
ax1.plot(x, ytrn, label = "training", color = 'peru')
ax1.legend()

# ax1.set_yticks(fontsize= 2)
# plt.title("Root Mean Squared Error by Degree of Fit", fontsize = 15) 
# ax1.xlabel("Degree of Fit", fontsize = 12)
# plt.xticks(fontsize = 9) 
ax1.set_ylabel("RMSE", fontsize = 12) 

# Mean Squared Error

yval2 = [71.788, 43.923, 45.478, 50.332, 55.898, 46.166, 47.222, 36.018, 
         35.967, 36.014, 36.012, 36.014
     ]
ytrn2 = [112.355, 82.151, 51.333, 32.56, 25.884, 23.034, 22.737, 20.433,
         20.448, 20.447, 20.447, 20.447
    ]


ax2.plot(x, yval2, label = "validation", color = 'g')
ax2.plot(x, ytrn2, label = "training", color = 'peru')

# labels
ax2.set_ylabel("MSE", fontsize = 12) 
ax2.set_xlabel("Degree of Fit")

plt.show()

# set the name of the figure
name_out = 'Error_v_FitDeg.png'
fig = fig.get_figure()
fig.savefig(figs_out + name_out, dpi = 900 )
plt.close(fig)

#%%

data = {'Degree of Fit':x,
        'Validation RMSE': yval,
        'Training RMSE': ytrn,
        'Validation MSE': yval2,
        'Training MSE': ytrn2}

df = pd.DataFrame(data)
print(df)

df.to_csv(out + 'Error_Levels_By_Fit.csv')