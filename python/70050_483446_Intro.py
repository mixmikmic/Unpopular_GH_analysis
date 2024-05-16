import numpy as np
print("numpy version: %s" % np.__version__)
x = np.array([[1, 2, 3], [4, 5, 6]])
x

import sklearn
print("scikit-learn version: %s" % sklearn.__version__)

import IPython
print("IPython version: %s" % IPython.__version__)

from scipy import sparse
# create a 2d numpy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print("Numpy array:\n%s" % eye)
# convert the numpy array to a scipy sparse matrix in CSR format
# only the non-zero entries are stores
sparse_matrix = sparse.csr_matrix(eye)
print("\nScipy sparse CSR matrix:\n%s" % sparse_matrix)

#matplotlib inline
import matplotlib.pyplot as plt
# Generate a sequence of integers
x = np.arange(20)
# create a second array using sinus
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker = "x")
plt.show()

import matplotlib
print("matplotlib version: %s" % matplotlib.__version__)

import pandas as pd
print("pandas version: %s" % pd.__version__)
# Create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location': ["New York", "Paris", "Berlin", "London"],
        'Age': [23, 13, 53, 33]
        }
data_pandas = pd.DataFrame(data)
data_pandas

