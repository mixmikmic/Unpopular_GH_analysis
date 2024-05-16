import random
import copy
import numpy as np

start_vect = [1,0,0] #doors

samples = 5000 #number of simulations to run

change, no_change = [],[]
for i in range(samples):
    
    #shuffle data
    vect = copy.copy(start_vect)
    random.shuffle(vect)

    #make choice
    choice = vect.pop(random.randint(0,2))
    no_change.append(choice) #outcome if do not change choice

    #show bad door
    try:
        bad = vect.pop(int(np.where(np.array(vect)==0)[0]))
    except:
        bad = vect.pop(0)

    change.append(vect) #outcome if change choice

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

plt.bar([0.5,1.5],[np.mean(change),np.mean(no_change)],width=1.0)
plt.xlim((0,3))
plt.ylim((0,1))
plt.ylabel('Proportion Correct Choice')
plt.xticks((1.0,2.0),['Change Choice', 'Do not chance choice'])

import scipy.stats as stats
obs = np.array([[np.sum(change), np.sum(no_change)], [samples, samples]])
print('Probability of choosing correctly if change choice: %0.2f' % np.mean(change))
print('Probability of choosing correctly if do not change choice: %0.2f' % np.mean(no_change))
print('Probability of difference arising from chance: %0.5f' % stats.chi2_contingency(obs)[1])

change, no_change = [],[]
for i in range(samples):
    #shuffle data
    vect = copy.copy(start_vect)
    random.shuffle(vect)

    #show bad door
    bad = vect.pop(int(np.where(np.array(vect)==0)[0][0]))

    #make choice
    choice = vect.pop(random.randint(0,1))
    no_change.append(choice)

    change.append(vect)

plt.bar([0.5,1.5],[np.mean(change),np.mean(no_change)],width=1.0)
plt.xlim((0,3))
plt.ylim((0,1))
plt.ylabel('Proportion Correct Choice')
plt.xticks((1.0,2.0),['Change Choice', 'Do not chance choice'])

obs = np.array([[np.sum(change), np.sum(no_change)], [samples, samples]])
print('Probability of choosing correctly if change choice: %0.2f' % np.mean(change))
print('Probability of choosing correctly if do not change choice: %0.2f' % np.mean(no_change))
print('Probability of difference arising from chance: %0.5f' % stats.chi2_contingency(obs)[1])

