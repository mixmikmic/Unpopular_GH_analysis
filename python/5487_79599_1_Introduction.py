#We will use Pandas Dataframe object to hold our data
#and to perform necessary manipulations to prepare the dataset
#for analysis 
import pandas as pd
#%matplotlib inline
get_ipython().magic('matplotlib inline')
#Iport matplotlib.pyplot for plotting results
import matplotlib.pyplot as plt
#Numpy will be used to perform numerical operations on arrays
#(calculate dot products, sums, exponentials, logarithms, find unique values)  
import numpy as np
#We will use scipy to calculate normal 
#distribution values
from scipy.stats import norm
#To set the working directory
import os as osvariable
#To read in csv file
from pandas import read_csv
#Lifelines is a survival analysis package. We will
#use its KaplanMeier curve plotting function,
#logrank_test and Cox proportional hazards fitter
#http://lifelines.readthedocs.org/en/latest/
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test   
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
#Import the statsmodels. We will use this to 
#fit linear functions to data, which will be 
#helpful to visually assess parametric fits
#http://statsmodels.sourceforge.net/
import statsmodels.api as st
#Genericlikelihood model is what we will use 
#to specify log-likelihood functions for survival
#models: Exponential (accelerated failure time (AFT), proportional hazards (PH)), 
#Weibull (AFT, PH), Gompertz (PH), Log-logistic (proportional odds (PO)), 
#Log-normal (AFT), Generalized Gamma (AFT) 
from statsmodels.base.model import GenericLikelihoodModel
#Import the functions that will be used to calculate the 
#generalized gamma function survival and its confidence
#intervals
#Gamma function
#http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.gamma.html
from scipy.special import gamma as gammafunction
#Lower regularized incomplete gamma function
#http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.gammainc.html
from scipy.special import gammainc
#Digamma function, which is used when taking the 
#derivative of the gamma function
from scipy.special import psi
#From mpmath library, we will use the meijer G function
#which is part of the derivative of the incomplete gamma function
#http://mpmath.googlecode.com/svn-history/r1229/trunk/doc/build/functions/hypergeometric.html
import mpmath
#from sympy library, we will use the DiracDelta function
#which is part of the derivative of the sign function which in turn
#is part of the generalized gamma function
#http://docs.sympy.org/dev/modules/functions/special.html
from sympy import DiracDelta

#set working directory:
osvariable.chdir('C:/----/----')
#Read in data
terrordata = read_csv('terrordata1.csv')
#Take a look at the dataset contents, print the first 5 observations
print(terrordata.head())
#Check the categorical variable values
print('Categorical variable values:')
print('Type values:',np.unique(terrordata['Type']))
print('Operating Peak Size values:',np.unique(terrordata['Operating Peak Size']))
print('Regime values:',np.unique(terrordata['Regime']))
print('Goal values:',np.unique(terrordata['Goal']))

#One of the entries for 'Type' is entered as 'Reigious'. This
#should be coded as 'R'
terrordata.loc[(terrordata['Type'] == 'Reigious'),['Type']] = 'R'
#Correct the 'Operating Peak Size' variables that are 
#entered incorrectly
terrordata.loc[(terrordata['Operating Peak Size'] == '10S'),['Operating Peak Size']] = '10s'
terrordata.loc[(terrordata['Operating Peak Size'] == '10c'),['Operating Peak Size']] = '10s'
terrordata.loc[(terrordata['Operating Peak Size'] == '1,00s'),['Operating Peak Size']] = '1,000s'
#One of the entries for 'Regime' is entered incorrectly as 'BF'
terrordata.loc[(terrordata['Regime'] == 'BF'),['Regime']] = 'NF'
#One of the entries for 'Goal' is entered incorrectly as 'TCs'
terrordata.loc[(terrordata['Goal'] == 'TCs'),['Goal']] = 'TC'
#Check the categorical variable values again
print(np.unique(terrordata['Type']))
print(np.unique(terrordata['Operating Peak Size']))
print(np.unique(terrordata['Regime']))
print(np.unique(terrordata['Goal']))

#Take a look at the unique values for categorical variables
#Check the categorical variable values
print('Categorical variable values:')
print('Type values:',np.unique(terrordata['Type']))
print('Operating Peak Size values:',np.unique(terrordata['Operating Peak Size']))
print('Regime values:',np.unique(terrordata['Regime']))
print('Goal values:',np.unique(terrordata['Goal']))

#Replace abbreviations with words to make reading tables easier
terrordata.loc[terrordata['Type'] == 'R',['Type']] = 'Religious'
terrordata.loc[terrordata['Type'] == 'LW',['Type']] = 'Left_wing'
terrordata.loc[terrordata['Type'] == 'N',['Type']] = 'Nationalist'
terrordata.loc[terrordata['Type'] == 'RW',['Type']] = 'Right_wing'

terrordata.loc[terrordata['Regime'] == 'F',['Regime']] = 'Free'
terrordata.loc[terrordata['Regime'] == 'PF',['Regime']] = 'Partly_free'
terrordata.loc[terrordata['Regime'] == 'NF',['Regime']] = 'Not_free'

terrordata.loc[terrordata['Goal'] == 'RC',['Goal']] = 'Regime_change'
terrordata.loc[terrordata['Goal'] == 'TC',['Goal']] = 'Territorial_change'
terrordata.loc[terrordata['Goal'] == 'PC',['Goal']] = 'Policy_change'
terrordata.loc[terrordata['Goal'] == 'E',['Goal']] = 'Empire'
terrordata.loc[terrordata['Goal'] == 'SR',['Goal']] = 'Social_revolution'
terrordata.loc[terrordata['Goal'] == 'SQ',['Goal']] = 'Status_Quo'

terrordata.loc[terrordata['Econ.'] == 'L',['Econ.']] = 'Low_income'
terrordata.loc[terrordata['Econ.'] == 'LM',['Econ.']] = 'Lower_middle_income'
terrordata.loc[terrordata['Econ.'] == 'UM',['Econ.']] = 'Upper_middle_income'
terrordata.loc[terrordata['Econ.'] == 'H',['Econ.']] = 'High_income'

terrordata.loc[terrordata['Reason'] == 'PO',['Reason']] = 'Policing'
terrordata.loc[terrordata['Reason'] == 'S',['Reason']] = 'Splintering'
terrordata.loc[terrordata['Reason'] == 'PT',['Reason']] = 'Politics'
terrordata.loc[terrordata['Reason'] == 'V',['Reason']] = 'Victory'
terrordata.loc[terrordata['Reason'] == 'MF',['Reason']] = 'Military_force'

#Now print the variable names
print(terrordata.columns)

#Create dummy variables for categorical variables
#Store dummy variables for each variable
sizevars = pd.get_dummies(terrordata['Operating Peak Size'])
econvars = pd.get_dummies(terrordata['Econ.'])
regimevars = pd.get_dummies(terrordata['Regime'])
typevars = pd.get_dummies(terrordata['Type'])
goalvars = pd.get_dummies(terrordata['Goal'])
reasonvars = pd.get_dummies(terrordata['Reason'])

#Add all dummy variables to the original dataset
for var in sizevars:
    terrordata[var] = sizevars[var]
for var in econvars:
    terrordata[var] = econvars[var]
for var in regimevars:
    terrordata[var] = regimevars[var]
for var in typevars:
    terrordata[var] = typevars[var]
for var in goalvars:
    terrordata[var] = goalvars[var]
for var in reasonvars:
    terrordata[var] = reasonvars[var]
    
#The dataset now includes all variables and their dummies
print(terrordata.columns)  

#Create the dataframe that we will use for analyses.
#Because we have categorical variables, we will leave 
#one dummy variable from each categorical variable out 
#as the reference case. Note that we are leaving
#variables for 'reason' out, since one of the categories
#of this variable ('not ended') matches the '0' value of the 
#'Event' variable 

#Reference categories that are left out are 
#'Regime_change', '10,000s', 'High_income'
#'Not_free', 'Left_wing'.
survivaldata = terrordata[['Territorial_change','Policy_change','Empire','Social_revolution','Status_Quo','1,000s','100s','10s','Low_income','Lower_middle_income','Upper_middle_income','Partly_free','Free','Nationalist','Religious','Right_wing']]    

#Add a constant term to the data
survivaldata = st.add_constant(survivaldata, prepend=False)

#Create the event variable. 'Ended' equals 1 if the terrorist group has 
#ended within the observation period and to 0 if it did not
eventvar = terrordata['Ended']

#Create the time variable. Time is in years and it is assumed that the minimum
#value it takes is 1
timevar = terrordata['Time']

