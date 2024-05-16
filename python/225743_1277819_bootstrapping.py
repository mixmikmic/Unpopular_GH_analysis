import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('dataset_Facebook.csv', delimiter=';')

#Boolean filtering/remove missing values
unpaid_likes = data[data['Paid']==0]['like']
paid_likes = data[data['Paid']==1]['like']
paid_likes = paid_likes.dropna()
unpaid_likes = unpaid_likes.dropna()

#Figure settings
sns.set(font_scale=1.65)
fig = plt.figure(figsize=(10,6))
fig.subplots_adjust(hspace=.5)    

#Plot top histogram
ax = fig.add_subplot(2, 1, 1)
ax = unpaid_likes.hist(range=(0, 1500),bins=50)
ax.set_xlim(0,800)
plt.xlabel('Likes (Unpaid)')
plt.ylabel('Frequency')

#Plot bottom histogram
ax2 = fig.add_subplot(2, 1, 2)
ax2 = paid_likes.hist(range=(0, 1500),bins=50)
ax2.set_xlim(0,800)

plt.xlabel('Likes (Paid)')
plt.ylabel('Frequency')

plt.show()
print('unpaid_mean: {}'.format(unpaid_likes.mean()))
print('paid_mean: {}'.format(paid_likes.mean()))

print('paid_size: {}'.format(len(paid_likes)))
print('unpaid_size: {}'.format(len(unpaid_likes)))

from sklearn.utils import resample
resample(paid_likes).head()

resample(paid_likes).head()

resample(paid_likes).head()

paid_bootstrap = []
for i in range(10000):
    np.random.seed(i)
    paid_bootstrap.append((resample(paid_likes)))
print(len(paid_bootstrap))

bootstrap_means = np.mean(paid_bootstrap, axis=1)
bootstrap_means

lower_bound = np.percentile(bootstrap_means, 2.5)
upper_bound = np.percentile(bootstrap_means, 97.5)

fig = plt.figure(figsize=(10,3))
ax = plt.hist(bootstrap_means, bins=30)

plt.xlabel('Likes (Paid)')
plt.ylabel('Frequency')
plt.axvline(lower_bound, color='r')
plt.axvline(upper_bound, color='r')
plt.show()

print('Lower bound: {}'.format(lower_bound))
print('Upper bound: {}'.format(upper_bound))

fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Paid', y='like', data=data, ci=95)
x = ['Paid Posts', 'Unpaid Posts']

plt.xticks([0, 1],x)
plt.ylabel('Likes')
plt.xlabel('')
plt.show() 

paid_bootstrap = []
for i in range(10000):
    np.random.seed(i)
    paid_bootstrap.append((resample(paid_likes)))

paid_bootstrap = np.mean(paid_bootstrap, axis=1)
paid_bootstrap

unpaid_bootstrap = []
for i in range(10000):
    np.random.seed(i)
    unpaid_bootstrap.append((resample(unpaid_likes)))
    
unpaid_bootstrap = np.mean(unpaid_bootstrap, axis=1)
unpaid_bootstrap

differences = paid_bootstrap - unpaid_bootstrap
lower_bound = np.percentile(differences, 2.5)
upper_bound = np.percentile(differences, 97.5)

fig = plt.figure(figsize=(10,3))
ax = plt.hist(differences, bins=30)

plt.xlabel('Difference in Likes')
plt.ylabel('Frequency')
plt.axvline(lower_bound, color='r')
plt.axvline(upper_bound, color='r')
plt.title('Bootstrapped Population (Difference Between 2 Groups)')
plt.show()

print('Lower bound: {}'.format(lower_bound))
print('Upper bound: {}'.format(upper_bound))

differences[differences <= 0].shape[0]

combined = np.concatenate((paid_likes, unpaid_likes), axis=0)

perms_paid = []
perms_unpaid = []

for i in range(10000):
    np.random.seed(i)
    perms_paid.append(resample(combined, n_samples = len(paid_likes)))
    perms_unpaid.append(resample(combined, n_samples = len(unpaid_likes)))
    
dif_bootstrap_means = (np.mean(perms_paid, axis=1)-np.mean(perms_unpaid, axis=1))
dif_bootstrap_means

fig = plt.figure(figsize=(10,3))
ax = plt.hist(dif_bootstrap_means, bins=30)

plt.xlabel('Difference in Likes')
plt.ylabel('Frequency')
plt.title('Bootstrapped Population (Combined data)')
plt.show()

obs_difs = (np.mean(paid_likes) - np.mean(unpaid_likes))
print('observed difference in means: {}'.format(obs_difs))

p_value = dif_bootstrap_means[dif_bootstrap_means >= obs_difs].shape[0]/10000
print('p-value: {}'.format(p_value))

fig = plt.figure(figsize=(10,3))
ax = plt.hist(dif_bootstrap_means, bins=30)

plt.xlabel('Difference in Likes')
plt.ylabel('Frequency')
plt.axvline(obs_difs, color='r')
plt.show()

