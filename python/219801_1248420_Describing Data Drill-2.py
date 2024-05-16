import pandas as pd
import numpy as np
import statistics as stat

names = ('Greg', 'Marcia', 'Peter', 'Jan', 'Bobby', 'Cindy', 'Oliver')
ages = np.array([14, 12, 11, 10, 8, 6, 8])
brady_bunch = pd.DataFrame(ages, columns=['Age'], index= names)
brady_bunch

#original mean
np.mean(brady_bunch['Age'])

#original median
np.median(brady_bunch['Age'])

#original mode
stat.mode(brady_bunch['Age'])

#original variance
np.var(brady_bunch['Age'])

#original standard deviation
np.std(brady_bunch['Age'])

#original standard error
np.std(brady_bunch['Age']) / np.sqrt(len(brady_bunch['Age'])-1)

brady_bunch.Age.describe()

#Change Cindy's birthday
brady_bunch.at['Cindy', 'Age'] = 7
brady_bunch

#Cindy Updated Birthday Mean
np.mean(brady_bunch['Age'])

#Cindy Updated Birthday Median
np.median(brady_bunch['Age'])

#Cindy Updated Birthday Mode
stat.mode(brady_bunch['Age'])

#Cindy Updated Birthday Variance
np.var(brady_bunch['Age'])

#Cindy Updated Birthday Standard Deviation
np.std(brady_bunch['Age'])

#Cindy Updated Birthday Standard Error
np.std(brady_bunch['Age']) / np.sqrt(len(brady_bunch['Age'])-1)

#Substitute Jessica for Oliver
names = ('Greg', 'Marcia', 'Peter', 'Jan', 'Bobby', 'Cindy', 'Jessica')
ages = np.array([14, 12, 11, 10, 8, 6, 1])
brady_bunch = pd.DataFrame(ages, columns=['Age'], index= names)
brady_bunch

#Jessica Substitute Mean
np.mean(brady_bunch['Age'])

#Jessica Substitute Median
np.median(brady_bunch['Age'])

#Jessica Substitute Mode - no unique mode

#Jessica Substitute Variance
np.var(brady_bunch['Age'])

#Jessica Substitute Standard Deviation
np.std(brady_bunch['Age'])

#Jessica Substitute Standard Error
np.std(brady_bunch['Age']) / np.sqrt(len(brady_bunch['Age'])-1)

#Question 5: Because SciPhi Phanatics is likely to not have an interest in a show like the
# Brady Bunch to begin with, I believe that is selction bias and can be removed from this 
# question. That being said, if we take the mean of the other 3, then I think that would be
# a more accurate representation - which is 20% as shown below.

np.mean([17, 20, 23])



