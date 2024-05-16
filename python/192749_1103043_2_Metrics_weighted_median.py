import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

N = 5
x = np.random.randint(low=1, high=100, size=N)
x

inv_x = 1.0/x
inv_x

w = inv_x/sum(inv_x)
w

idxs = np.argsort(w)
sorted_w = w[idxs]
sorted_w

sorted_w_cumsum = np.cumsum(sorted_w)
plt.plot(sorted_w_cumsum); plt.show()
print ('sorted_w_cumsum: ', sorted_w_cumsum)

idx = np.where(sorted_w_cumsum>0.5)[0][0]
idx

pos = idxs[idx]
x[pos]

print('Data: ', x)
print('Sorted data: ', np.sort(x))
print('Weighted median: %d, Median: %d' %(x[pos], np.median(x)))

