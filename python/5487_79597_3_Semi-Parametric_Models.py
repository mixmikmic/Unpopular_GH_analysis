#CoxPHFitter takes a dataframe that includes time and event 
#variables
survivaldata_time_event = survivaldata.copy()
#Remove the constant term, since the Cox proportional hazards model does not estimate a parameter for the constant term
survivaldata_time_event = survivaldata_time_event.iloc[:,:-1]
#Add the time and censoring variables to the dataframe
survivaldata_time_event['Time'] = timevar
survivaldata_time_event['Event'] = eventvar
#Create the CoxPHFitter object
cf = CoxPHFitter()
#Fit the model
cf.fit(survivaldata_time_event, 'Time', event_col='Event',include_likelihood = True)
#Print the summary from the cox-proportional hazard model
cf.print_summary()

#This is the function that returns scaled schoenfeld residuals
#y is the 'time' variable, 'X' is the n by p data matrix, 
#'params' is the parameter values from the Cox proportional hazard model
#'covar' is the variance-covariance matrix from the Cox proportional
#hazard model
def schoenfeldresiduals(y,X,event,params,covar):
    #Create a dataframe to hold the scaled Schoenfeld residuals
    schoenfeldresiduals = pd.DataFrame(columns=[X.columns])
    #Create a dataframe to hold the 'Time variable'
    schoenfeldtime = pd.DataFrame(columns=['Time'])
    #Add the 'Time' variable to the data matrix 'X'. This will be
    #useful to select units still at risk of event
    X['Time'] = y
    #Add the 'event' variable to the data matrix 'X'. This will be
    #useful to select units who experienced the event
    X['Eventoccured'] = event
    #Sort 'X' based on time (ascending order)
    X = X.sort(['Time'],axis = 0)
    #Get the number of units
    numberofunits = len(X)
    #Set the counter to zero
    counter = 0
    #Get the number of units that experienced the event
    numberofevents = np.sum(event)
    #For each unit, calculate the residuals if they experienced the event
    for patientindex in xrange(numberofunits):
        if X['Eventoccured'].iloc[patientindex] == 1:
            currenttime = X['Time'].iloc[patientindex]
            #Sum of the hazards for all the observations still at risk
            sumhazards = np.sum(np.exp(np.dot(X.loc[X['Time'] >= currenttime].iloc[:,:len(X.columns) - 2],params)))
            #Calculate the probability of event for each unit still at risk
            probabilityofdeathall = np.ravel(np.exp(np.dot(X.loc[X['Time'] >= currenttime].iloc[:,:len(X.columns) - 2],params)) / sumhazards)         
            #Calculate the expected covariate values
            expectedcovariatevalues = np.dot(probabilityofdeathall,X.loc[(X['Time'] >= currenttime)].iloc[:,:len(X.columns) - 2])
            #Get Schoenfeld residuals as the difference between the current unit's covariate values and the 
            #expected covariate values calculated from all units at risk
            residuals = X.iloc[patientindex,:len(X.columns) - 2] - expectedcovariatevalues
            #Scale the residuals by the variance-covariance matrix of model parameters
            scaledresiduals = numberofevents * np.dot(covar,residuals)
            #Add the scaled residuals to the dataframe for residuals
            schoenfeldresiduals.loc[counter]= scaledresiduals
            #Add the current time for the current unit. This can be used to regress scaled residuals against time
            schoenfeldtime.loc[counter] = currenttime          
            counter = counter + 1
    schoenfeldresiduals['Time'] = schoenfeldtime
    return schoenfeldresiduals

residuals = schoenfeldresiduals(timevar,survivaldata.iloc[:,:-1],eventvar,cf.hazards_.transpose(),cf.covar())   

#Start plotting the scaled residuals against time
plt.figure()
plt.scatter(residuals['Time'],residuals['Territorial_change'])
plt.title('Territorial_change')
#Fit a linear model to scaled residuals and time data, after adding a constant
#to 'Time'
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Territorial_change'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

#Repeat for residuals from other variables
plt.figure()
plt.scatter(residuals['Time'],residuals['Policy_change'])
plt.title('Policy_change')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Policy_change'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Empire'])
plt.title('Empire')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Empire'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Social_revolution'])
plt.title('Social_revolution')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Social_revolution'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Status_Quo'])
plt.title('Status_Quo')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Status_Quo'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['1,000s'])
plt.title('1,000s')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['1,000s'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['100s'])
plt.title('100s')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['100s'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['10s'])
plt.title('10s')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['10s'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Low_income'])
plt.title('Low_income')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Low_income'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Lower_middle_income'])
plt.title('Lower_middle_income')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Lower_middle_income'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Upper_middle_income'])
plt.title('Upper_middle_income')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Upper_middle_income'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Partly_free'])
plt.title('Partly_free')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Partly_free'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Free'])
plt.title('Free')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Free'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Nationalist'])
plt.title('Nationalist')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Nationalist'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Religious'])
plt.title('Religious')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Religious'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

plt.figure()
plt.scatter(residuals['Time'],residuals['Right_wing'])
plt.title('Right_wing')
X = st.add_constant(residuals['Time'],prepend = False)
linearmodel = st.OLS(residuals['Right_wing'],X)
linearmodelresults = linearmodel.fit()
plt.plot(residuals['Time'],linearmodelresults.fittedvalues,'g')
print(linearmodelresults.summary())

#Start plotting the log(-log(S(t))) against log(t). First, get the Kaplan-Meier estimates for S(t).
plt.figure()
kmf.fit(timevar[tens],event_observed = eventvar[tens],label = "Tens")
#Plot the log(-log(S(t))) against log(t)
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Tens'])),linewidth = 2.0,label = "Tens")
#Fit a linear equation for easier visual assessment of the plots
y = np.log(-np.log(kmf.survival_function_['Tens'].iloc[1:len(kmf.survival_function_)]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'b--',linewidth = 2.0)

#Repeat for groups of different peak operating sizes
kmf.fit(timevar[hundreds],event_observed = eventvar[hundreds],label = "Hundreds")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Hundreds'])),linewidth = 2.0,label = "Hundreds")
y = np.log(-np.log(kmf.survival_function_['Hundreds'].iloc[1:len(kmf.survival_function_)]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

kmf.fit(timevar[thousands],event_observed = eventvar[thousands],label = "Thousands")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Thousands'])),linewidth = 2.0,label = "Thousands")
y = np.log(-np.log(kmf.survival_function_['Thousands'].iloc[5:len(kmf.survival_function_)]))
X = np.log(kmf.survival_function_.index.values[5:len(kmf.survival_function_.index.values)])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'r--',linewidth = 2.0)

kmf.fit(timevar[tenthousands],event_observed = eventvar[tenthousands],label = "Ten thousands")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Ten thousands'])),linewidth = 2.0,label = "Ten thousands")
y = np.log(-np.log(kmf.survival_function_['Ten thousands'].iloc[3:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[3:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'c--',linewidth = 2.0)

plt.legend(loc = 'lower right')
plt.ylabel('ln(-ln(S(t)))')
plt.xlabel('ln(Time)')
plt.title('Operating Peak Size')

#Repeat for groups operating against countries with different income levels
plt.figure()
kmf.fit(timevar[low],event_observed = eventvar[low],label = "Low")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Low'])),linewidth = 2.0,label = "Low")
y = np.log(-np.log(kmf.survival_function_['Low'].iloc[1:len(kmf.survival_function_)]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'b--',linewidth = 2.0)

kmf.fit(timevar[lowermiddle],event_observed = eventvar[lowermiddle],label = "Lower middle")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Lower middle'])),linewidth = 2.0,label = "Lower middle")
y = np.log(-np.log(kmf.survival_function_['Lower middle'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

kmf.fit(timevar[uppermiddle],event_observed = eventvar[uppermiddle],label = "Upper middle")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Upper middle'])),linewidth = 2.0,label = "Upper middle")
y = np.log(-np.log(kmf.survival_function_['Upper middle'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'r--',linewidth = 2.0)

kmf.fit(timevar[high],event_observed = eventvar[high],label = "High")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['High'])),linewidth = 2.0,label = "High")
y = np.log(-np.log(kmf.survival_function_['High'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'c--',linewidth = 2.0)

plt.legend(loc = 'lower right')
plt.ylabel('ln(-ln(S(t)))')
plt.xlabel('ln(Time)')
plt.title('Economy')

#Repeat for groups operating against countries with different freedom index values
plt.figure()
kmf.fit(timevar[free],event_observed = eventvar[free],label = "Free")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Free'])),linewidth = 2.0,label = "Free")
y = np.log(-np.log(kmf.survival_function_['Free'].iloc[1:len(kmf.survival_function_)]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'b--',linewidth = 2.0)

kmf.fit(timevar[partlyfree],event_observed = eventvar[partlyfree],label = "Partly free")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Partly free'])),linewidth = 2.0,label = "Partly free")
y = np.log(-np.log(kmf.survival_function_['Partly free'].iloc[1:len(kmf.survival_function_)]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

kmf.fit(timevar[notfree],event_observed = eventvar[notfree],label = "Not free")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Not free'])),linewidth = 2.0,label = "Not free")
y = np.log(-np.log(kmf.survival_function_['Not free'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'r--',linewidth = 2.0)

plt.legend(loc = 'lower right')
plt.ylabel('ln(-ln(S(t)))')
plt.xlabel('ln(Time)')
plt.title('Regime')

#Repeat for groups with different ideologies
plt.figure()
kmf.fit(timevar[nationalist],event_observed = eventvar[nationalist],label = "Nationalist")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Nationalist'])),linewidth = 2.0,label = "Nationalist")
y = np.log(-np.log(kmf.survival_function_['Nationalist'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'b--',linewidth = 2.0)

kmf.fit(timevar[religious],event_observed = eventvar[religious],label = "Religious")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Religious'])),linewidth = 2.0,label = "Religious")
y = np.log(-np.log(kmf.survival_function_['Religious'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

kmf.fit(timevar[right_wing],event_observed = eventvar[right_wing],label = "Right wing")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Right wing'])),linewidth = 2.0,label = "Right wing")
y = np.log(-np.log(kmf.survival_function_['Right wing'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'r--',linewidth = 2.0)

kmf.fit(timevar[left_wing],event_observed = eventvar[left_wing],label = "Left wing")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Left wing'])),linewidth = 2.0,label = "Left wing")
y = np.log(-np.log(kmf.survival_function_['Left wing'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'c--',linewidth = 2.0)

plt.legend(loc = 'lower right')
plt.ylabel('ln(-ln(S(t)))')
plt.xlabel('ln(Time)')
plt.title('Type')

#Repeat for groups with different goals
plt.figure()
kmf.fit(timevar[territorial],event_observed = eventvar[territorial],label = "Territorial change")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Territorial change'])),linewidth = 2.0,label = "Territorial change")
y = np.log(-np.log(kmf.survival_function_['Territorial change'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'b--',linewidth = 2.0)

kmf.fit(timevar[policy],event_observed = eventvar[policy],label = "Policy change")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Policy change'])),linewidth = 2.0,label = "Policy change")
y = np.log(-np.log(kmf.survival_function_['Policy change'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'g--',linewidth = 2.0)

kmf.fit(timevar[empire],event_observed = eventvar[empire],label = "Empire")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Empire'])),linewidth = 2.0,label = "Empire")
y = np.log(-np.log(kmf.survival_function_['Empire'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'r--',linewidth = 2.0)

kmf.fit(timevar[social],event_observed = eventvar[social],label = "Social revolution")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Social revolution'])),linewidth = 2.0,label = "Social revolution")
y = np.log(-np.log(kmf.survival_function_['Social revolution'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'c--',linewidth = 2.0)

kmf.fit(timevar[status],event_observed = eventvar[status],label = "Status Quo")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Status Quo'])),linewidth = 2.0,label = "Status Quo")
y = np.log(-np.log(kmf.survival_function_['Status Quo'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'m--',linewidth = 2.0)

kmf.fit(timevar[regime],event_observed = eventvar[regime],label = "Regime change")
plt.plot(np.log(kmf.survival_function_.index.values),np.log(-np.log(kmf.survival_function_['Regime change'])),linewidth = 2.0,label = "Regime change")
y = np.log(-np.log(kmf.survival_function_['Regime change'].iloc[1:len(kmf.survival_function_)-1]))
X = np.log(kmf.survival_function_.index.values[1:len(kmf.survival_function_.index.values)-1])
X = st.add_constant(X, prepend=False)
linearmodel = st.OLS(y,X)
linearmodelresults = linearmodel.fit()
plt.plot(X[:,[0]],linearmodelresults.fittedvalues,'y--',linewidth = 2.0)

plt.legend(loc = 'lower right')
plt.ylabel('ln(-ln(S(t)))')
plt.xlabel('ln(Time)')
plt.title('Type')

