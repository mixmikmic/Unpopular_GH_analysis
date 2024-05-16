import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

# NumPy as arrays can be passed directly to MLlib
denseVec1 = np.array([1, 2, 3])
denseVec2 = Vectors.dense([1, 2, 3])

sparseVec1 = Vectors.sparse(4, {0:1.0, 2:2.0})
sparseVec2 = Vectors.sparse(4, [0, 2], [1.0, 2.0])
print sparseVec1
print sparseVec2

lp1 = LabeledPoint(12.0, np.array([45.3, 4.1, 7.0]))
print lp1, type(lp1)

lp2 = LabeledPoint(1.0, Vectors.dense([4.4, 1.1, -23.0]))
print lp2, type(lp2)

