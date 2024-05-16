import sklearn.datasets
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# default parameters
sns.set()

# clear string
def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string)
    return string

# because of sklean.datasets read a document as a single element
# so we want to split based on new line
def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        # python3, if python2, just remove list()
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget

trainset = sklearn.datasets.load_files(container_path = 'local', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset)

from sklearn.cross_validation import train_test_split

# default colors from seaborn
current_palette = sns.color_palette(n_colors = len(trainset.filenames))

# visualize 5% of our data
_, texts, _, labels = train_test_split(trainset.data, trainset.target, test_size = 0.05)

# bag-of-word
bow = CountVectorizer().fit_transform(texts)

#tf-idf, must get from BOW first
tfidf = TfidfTransformer().fit_transform(bow)

#hashing, default n_features, probability cannot divide by negative
hashing = HashingVectorizer(non_negative = True).fit_transform(texts)

# size of figure is 1000 x 500 pixels
plt.figure(figsize = (15, 5))

plt.subplot(1, 2, 1)
composed = PCA(n_components = 2).fit_transform(bow.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('PCA')

plt.subplot(1, 2, 2)
composed = TSNE(n_components = 2).fit_transform(bow.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('TSNE')

plt.show()

# size of figure is 1000 x 500 pixels
plt.figure(figsize = (15, 5))

plt.subplot(1, 2, 1)
composed = PCA(n_components = 2).fit_transform(tfidf.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('PCA')

plt.subplot(1, 2, 2)
composed = TSNE(n_components = 2).fit_transform(tfidf.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('TSNE')

plt.show()

# size of figure is 1000 x 500 pixels
plt.figure(figsize = (15, 5))

plt.subplot(1, 2, 1)
composed = PCA(n_components = 2).fit_transform(hashing.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('PCA')

plt.subplot(1, 2, 2)
composed = TSNE(n_components = 2).fit_transform(hashing.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('TSNE')

plt.show()



