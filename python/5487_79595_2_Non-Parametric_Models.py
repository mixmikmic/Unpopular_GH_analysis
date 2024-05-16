#Create a KaplanMeier object, imported from lifelines
kmf = KaplanMeierFitter()
#Calculate the K-M curve for all groups
kmf.fit(timevar,event_observed = eventvar,label = "All groups")
#Plot the curve and assign labels
kmf.plot()
plt.ylabel('Probability of ending')
plt.xlabel('Years since start')
plt.title('Kaplan-Meier Curve')
#Print median survival time
print('Median survival',kmf.median_)

#Create boolean arrays to specify which groups will be included each K-M plot
tens = (survivaldata['10s'] == 1)
hundreds = (survivaldata['100s'] == 1)
thousands = (survivaldata['1,000s'] == 1)
tenthousands = ((survivaldata['10s'] == 0) & (survivaldata['100s'] == 0) & (survivaldata['1,000s'] == 0))
#Start a new plot
plt.figure()
ax = plt.subplot(1,1,1)
#Fit the K-M curve to observations for which Peak Operating Size = 10s
kmf.fit(timevar[tens],event_observed = eventvar[tens],label = "10s")
print('Median survival: 10s',kmf.median_)
plot1 = kmf.plot(ax = ax)
#Fit the K-M curve to observations for which Peak Operating Size = 100s
kmf.fit(timevar[hundreds],event_observed = eventvar[hundreds],label = "100s")
print('Median survival: 100s',kmf.median_)
plot2 = kmf.plot(ax = plot1)
#Fit the K-M curve to observations for which Peak Operating Size = 1,000s
kmf.fit(timevar[thousands],event_observed = eventvar[thousands],label = "1,000s")
print('Median survival: 1,000s',kmf.median_)
plot3 = kmf.plot(ax = plot2)
#Fit the K-M curve to observations for which Peak Operating Size = 10,000s
kmf.fit(timevar[tenthousands],event_observed = eventvar[tenthousands],label = "10,000s")
print('Median survival: 10,000s',kmf.median_)
plot4 = kmf.plot(ax = plot3)
plt.title('Probability of survival for terror groups with different operating peak size')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.yticks(np.linspace(0,1,11))
twoplusgroups_logrank = multivariate_logrank_test(terrordata['Time'],terrordata['Operating Peak Size'],terrordata['Ended'],alpha = 0.95)
twoplusgroups_logrank.print_summary()

#Create boolean arrays to specify which groups will be included each K-M plot
low = (survivaldata['Low_income'] == 1)
lowermiddle = (survivaldata['Lower_middle_income'] == 1)
uppermiddle = (survivaldata['Upper_middle_income'] == 1)
high = ((survivaldata['Low_income'] == 0) & (survivaldata['Lower_middle_income'] == 0) & (survivaldata['Upper_middle_income'] == 0))
#Start a new plot
plt.figure()
ax = plt.subplot(1,1,1)
#Fit the K-M curve to observations for which Economy = low
kmf.fit(timevar[low],event_observed = eventvar[low],label = "Low income")
print('Median survival: Low income',kmf.median_)
plot1 = kmf.plot(ax = ax)
#Fit the K-M curve to observations for which Economy = Lower middle income
kmf.fit(timevar[lowermiddle],event_observed = eventvar[lowermiddle],label = "Lower middle income")
print('Median survival: Lower middle income',kmf.median_)
plot2 = kmf.plot(ax = plot1)
#Fit the K-M curve to observations for which Economy = Upper middle income
kmf.fit(timevar[uppermiddle],event_observed = eventvar[uppermiddle],label = "Upper middle income")
print('Median survival: Upper middle income',kmf.median_)
plot3 = kmf.plot(ax = plot2)
#Fit the K-M curve to observations for which Economy = High income
kmf.fit(timevar[high],event_observed = eventvar[high],label = "High income")
print('Median survival: High income',kmf.median_)
plot4 = kmf.plot(ax = plot3)
plt.title('Probability of survival for terror groups in countries with different income levels')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.yticks(np.linspace(0,1,11))
twoplusgroups_logrank = multivariate_logrank_test(terrordata['Time'],terrordata['Econ.'],terrordata['Ended'],alpha = 0.95)
twoplusgroups_logrank.print_summary()

#Create boolean arrays to specify which groups will be included each K-M plot
free = (survivaldata['Free'] == 1)
partlyfree = (survivaldata['Partly_free'] == 1)
notfree = ((survivaldata['Free'] == 0) & (survivaldata['Partly_free'] == 0))
#Start a new plot
plt.figure()
ax = plt.subplot(1,1,1)
#Fit the K-M curve to observations for which regime = Free
kmf.fit(timevar[free],event_observed = eventvar[free],label = "Free")
print('Median survival: Free',kmf.median_)
plot1 = kmf.plot(ax = ax)
#Fit the K-M curve to observations for which regime = Partly_free
kmf.fit(timevar[partlyfree],event_observed = eventvar[partlyfree],label = "Partly Free")
print('Median survival: Partly Free',kmf.median_)
plot2 = kmf.plot(ax = plot1)
#Fit the K-M curve to observations for which regime = Not_free
kmf.fit(timevar[notfree],event_observed = eventvar[notfree],label = "Not Free")
print('Median survival: Not Free',kmf.median_)
kmf.plot(ax = plot2)
plt.title('Probability of survival for terror groups in different regime types')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.yticks(np.linspace(0,1,11))
twoplusgroups_logrank = multivariate_logrank_test(terrordata['Time'],terrordata['Regime'],terrordata['Ended'],alpha = 0.95)
twoplusgroups_logrank.print_summary()

#Create boolean arrays to specify which groups will be included each K-M plot
nationalist = (survivaldata['Nationalist'] == 1)
religious = (survivaldata['Religious'] == 1)
right_wing = (survivaldata['Right_wing'] == 1)
left_wing = ((survivaldata['Nationalist'] == 0) & (survivaldata['Religious'] == 0) & (survivaldata['Right_wing'] == 0))
#Start a new plot
plt.figure()
ax = plt.subplot(1,1,1)
#Fit the K-M curve to observations for which Type = Nationalist
kmf.fit(timevar[nationalist],event_observed = eventvar[nationalist],label = "Nationalist")
print('Median survival: Nationalist',kmf.median_)
plot1 = kmf.plot(ax = ax)
#Fit the K-M curve to observations for which Type = Religious
kmf.fit(timevar[religious],event_observed = eventvar[religious],label = "Religious")
print('Median survival: Religious',kmf.median_)
plot2 = kmf.plot(ax = plot1)
#Fit the K-M curve to observations for which Type = Right_wing
kmf.fit(timevar[right_wing],event_observed = eventvar[right_wing],label = "Right wing")
print('Median survival: Right wing',kmf.median_)
plot3 = kmf.plot(ax = plot2)
#Fit the K-M curve to observations for which Type = Left_wing
kmf.fit(timevar[left_wing],event_observed = eventvar[left_wing],label = "Left wing")
print('Median survival: Left wing',kmf.median_)
plot4 = kmf.plot(ax = plot3)
plt.title('Probability of survival for terror groups with different ideologies')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.yticks(np.linspace(0,1,11))
twoplusgroups_logrank = multivariate_logrank_test(terrordata['Time'],terrordata['Type'],terrordata['Ended'],alpha = 0.95)
twoplusgroups_logrank.print_summary()

#Create boolean arrays to specify which groups will be included each K-M plot
territorial = (survivaldata['Territorial_change'] == 1)
policy = (survivaldata['Policy_change'] == 1)
empire = (survivaldata['Empire'] == 1)
social = (survivaldata['Social_revolution'] == 1)
status = (survivaldata['Status_Quo'] == 1)
regime = (survivaldata['Territorial_change'] == 0) & (survivaldata['Policy_change'] == 0) & (survivaldata['Empire'] == 0) & (survivaldata['Social_revolution'] == 0) & (survivaldata['Status_Quo'] == 0)
#Start a new plot
plt.figure()
ax = plt.subplot(1,1,1)
#Fit the K-M curve to observations for which Goal = Territorial_change
kmf.fit(timevar[territorial],event_observed = eventvar[territorial],label = "Territorial change")
print('Median survival: Territorial change',kmf.median_)
plot1 = kmf.plot(ax = ax)
#Fit the K-M curve to observations for which Goal = Policy_change
kmf.fit(timevar[policy],event_observed = eventvar[policy],label = "Policy change")
print('Median survival: Policy change',kmf.median_)
plot2 = kmf.plot(ax = plot1)
#Fit the K-M curve to observations for which Goal = Empire
kmf.fit(timevar[empire],event_observed = eventvar[empire],label = "Empire")
print('Median survival: Empire',kmf.median_)
plot3 = kmf.plot(ax = plot2)
#Fit the K-M curve to observations for which Goal = Social_revolution
kmf.fit(timevar[social],event_observed = eventvar[social],label = "Social revolution")
print('Median survival: Social revolution',kmf.median_)
plot4 = kmf.plot(ax = plot3)
#Fit the K-M curve to observations for which Goal = Status Quo
kmf.fit(timevar[status],event_observed = eventvar[status],label = "Status_Quo")
print('Median survival: Status_Quo',kmf.median_)
plot5 = kmf.plot(ax = plot4)
#Fit the K-M curve to observations for which Goal = Regime_change
kmf.fit(timevar[regime],event_observed = eventvar[regime],label = "Regime change")
print('Median survival: Regime change',kmf.median_)
plot6 = kmf.plot(ax = plot5)
plt.title('Probability of survival for terror groups with different goals')
plt.xlabel('Years since start of group')
plt.ylabel('Probability of ending')
plt.yticks(np.linspace(0,1,11))
twoplusgroups_logrank = multivariate_logrank_test(terrordata['Time'],terrordata['Goal'],terrordata['Ended'],alpha = 0.95)
twoplusgroups_logrank.print_summary()

