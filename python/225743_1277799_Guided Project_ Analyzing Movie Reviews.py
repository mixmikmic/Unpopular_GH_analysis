import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

movies = pd.read_csv('fandango_score_comparison.csv')
movies.head()

mc = movies['Metacritic_norm_round']
fd = movies['Fandango_Stars']

plt.hist(mc, 5)
plt.show()

plt.hist(fd, 5)
plt.show()

mean_fd = fd.mean()
mean_mc = mc.mean()
median_fd = fd.median()
median_mc = mc.median()
std_fd = fd.std()
std_mc = mc.std()

print("means", mean_fd, mean_mc)
print("medians",median_fd, median_mc)
print("std_devs",std_fd, std_mc)

plt.scatter(fd, mc)
plt.show()

movies['fm_diff'] = fd - mc
movies['fm_diff'] = np.absolute(movies['fm_diff'])
dif_sort = movies['fm_diff'].sort_values(ascending=False)

movies.sort_values(by='fm_diff', ascending = False).head(5)

import scipy.stats as sci

r, pearsonr = sci.pearsonr(mc, fd)
print(r)
print(pearsonr)

m, b, r, p, stderr = sci.linregress(mc, fd)

#Fit into a line, y = mx+b where x is 3.
pred_3 = m*3 + b
pred_3

pred_1 = m*1 + b
print(pred_1)
pred_5 = m*5 + b
print(pred_5)

x_pred = [1.0, 5.0]
y_pred = [3.89708499687, 4.28632930877]

plt.scatter(fd, mc)
plt.plot(x_pred, y_pred)



plt.show()



