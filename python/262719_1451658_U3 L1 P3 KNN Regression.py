import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

music = pd.DataFrame()
music['duration'] = [184, 134, 243, 186, 122, 197, 294, 382, 102, 264, 
                     205, 110, 307, 110, 397, 153, 190, 192, 210, 403,
                     164, 198, 204, 253, 234, 190, 182, 401, 376, 102]
music['loudness'] = [18, 34, 43, 36, 22, 9, 29, 22, 10, 24, 
                     20, 10, 17, 51, 7, 13, 19, 12, 21, 22,
                     16, 18, 4, 23, 34, 19, 14, 11, 37, 42]
music['bpm'] = [105, 90, 78, 75, 120, 110, 80, 100, 105, 60,
                  70, 105, 95, 70, 90, 105, 70, 75, 102, 100,
                  100, 95, 90, 80, 90, 80, 100, 105, 70, 65]

from sklearn import neighbors

# Build our model.
knn = neighbors.KNeighborsRegressor(n_neighbors=10)
X = pd.DataFrame(music.loudness)
Y = music.bpm
knn.fit(X, Y)

# Set up our prediction line. 
T = np.arange(0, 50, 0.1)[:, np.newaxis]

# Trailing underscores are a common convention for prediction.
Y_ = knn.predict(T)
plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('K=10, Unweighted')
plt.figure(figsize=(20,10))
plt.show()

# Run the same model, this time with weights.
knn_w = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
X = pd.DataFrame(music.loudness)
Y = music.bpm
knn_w.fit(X, Y)

# Set up our prediction line.
T = np.arange(0, 50, 0.1)[:, np.newaxis]

Y_ = knn_w.predict(T)

plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('K=10, Weighted')
plt.show()

from sklearn.model_selection import cross_val_score
score = cross_val_score(knn, X, Y, cv=5)
print('Unweighted Accuracy: %0.2f (+/-%0.2f)'%(score.mean(), score.std() * 2))
score_w = cross_val_score(knn_w, X, Y, cv=5)
print('Weighted Accuracy: %0.2f (+/-%0.2f)'%(score_w.mean(), score_w.std() * 2))

# Changed the amount of neighbors.
knn = neighbors.KNeighborsRegressor(n_neighbors=15)
X = pd.DataFrame(music.duration) 
Y = music.bpm
knn.fit(X, Y)

# Set up our prediction line.
T = np.arange(0, 50, 0.1)[:, np.newaxis]

# Trailing underscores are a common convention for a prediction.
Y_ = knn.predict(T)

plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('K=5, Unweighted')
plt.show()

# Run the same model, this time with weights.
knn_w = neighbors.KNeighborsRegressor(n_neighbors=15, weights='distance')
X = pd.DataFrame(music.loudness)
Y = music.bpm
knn_w.fit(X, Y)

# Set up our prediction line.
T = np.arange(0, 50, 0.1)[:, np.newaxis]

Y_ = knn_w.predict(T)

plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('K=10, Weighted')
plt.show()

score = cross_val_score(knn, X, Y, cv=5)
print("Unweighted Accuracy (duration): %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
score_w = cross_val_score(knn_w, X, Y, cv=5)
print('Weighted Accuracy: %0.2f (+/-%0.2f)'%(score_w.mean(), score_w.std() * 2))

# Add Duration and loudness. 
knn = neighbors.KNeighborsRegressor(n_neighbors=15)
X = np.array(music.ix[:, 0:2]) 
Y = music.bpm
knn.fit(X, Y)
score = cross_val_score(knn, X, Y, cv=5)
print("Unweighted Accuracy (Loudness/Duration): %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
# Add weights to duration and loudness.
knn_w = neighbors.KNeighborsRegressor(n_neighbors=15, weights='distance')
X = np.array(music.ix[:, 0:2]) 
Y = music.bpm
knn.fit(X, Y)
score = cross_val_score(knn_w, X, Y, cv=5)
print('Weighted Accuracy: %0.2f (+/-%0.2f)'%(score_w.mean(), score_w.std() * 2))



