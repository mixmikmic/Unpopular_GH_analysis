get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F

#Prepare the data.
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from numpy import linalg as LA

dtype = torch.FloatTensor

images = np.load("./data/images.npy")
labels = np.load("./data/labels.npy")

images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

images = images - images.mean()
images = images/images.std() 

train_seqs = images[0:40000]
val_seqs = images[40000:50000]

train_labels = labels[0:40000]
cv_labels = labels[40000:50000]

# A nn model with 2 hidden layers
HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS, H = 26, 26, 5, 5000, 300
learning_rate = 0.001

class TwoLayerNN(torch.nn.Module):
    def __init__(self, D_in, D_out, layers):
        super(TwoLayerNN, self).__init__()
        #self.Linear = torch.nn.Linear(D_in, D_out)
        self.hidden_layer_count = layers
        self.Linear1 = torch.nn.Linear(D_in, H)
        self.middleLinear = torch.nn.Linear(H, H)
        self.Linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h = self.Linear1(x)
        h_relu = F.relu(h, inplace=False)
        for i in range(self.hidden_layer_count):
            h_middle = self.middleLinear(h_relu)
            h_middle_relu = F.relu(h_middle, inplace = False)
        y_pred = self.Linear2(h_middle_relu)
        return y_pred
        

model = TwoLayerNN(HEIGHT * WIDTH, NUM_CLASSES, 2)

optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

def train(batch_size):
    model.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]

def accuracy(y, y_hat):
    count = 0
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            count += 1
    return count/y.shape[0]

import random
def approx_train_accuracy():
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = train_seqs[i].astype(np.float32)
    y = train_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    lst = list(model.parameters())
        
    for i in range(1000):
        h1 = x[i].dot(lst[0].data.numpy().transpose()) + lst[1].data.numpy()
        h1_relu = np.maximum(0.0, h1)
        h2 = h1_relu.dot(lst[2].data.numpy().transpose()) + lst[3].data.numpy()
        h2_relu = np.maximum(0.0, h2)
        y_pred = h2_relu.dot(lst[4].data.numpy().transpose()) + lst[5].data.numpy()
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    y_hat = np.empty(1000)

    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = val_seqs[i].astype(np.float32)
    y = cv_labels[i].astype(np.int)
    
    
    lst = list(model.parameters())
    for i in range(1000):
        h1 = x[i].dot(lst[0].data.numpy().transpose()) + lst[1].data.numpy()
        h1_relu = np.maximum(0.0, h1)
        h2 = h1_relu.dot(lst[2].data.numpy().transpose()) + lst[3].data.numpy()
        h2_relu = np.maximum(0.0, h2)
        y_pred = h2_relu.dot(lst[4].data.numpy().transpose()) + lst[5].data.numpy()
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc

train_accs, val_accs = [], []
batch_size = 300
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))

import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS, H = 26, 26, 5, 5000, 100
learning_rate = 0.001
    
model2 = TwoLayerNN(HEIGHT * WIDTH, NUM_CLASSES, 3)
optimizer = torch.optim.Adam(model2.parameters(), lr= learning_rate)
def train(batch_size):
    model2.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model2(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]

import random
def approx_train_accuracy():
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = train_seqs[i].astype(np.float32)
    y = train_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    lst = list(model2.parameters())
        
    for i in range(1000):
        h1 = x[i].dot(lst[0].data.numpy().transpose()) + lst[1].data.numpy()
        h1_relu = np.maximum(0.0, h1)
        h2 = h1_relu.dot(lst[2].data.numpy().transpose()) + lst[3].data.numpy()
        h2_relu = np.maximum(0.0, h2)
        y_pred = h2_relu.dot(lst[4].data.numpy().transpose()) + lst[5].data.numpy()
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    y_hat = np.empty(1000)

    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = val_seqs[i].astype(np.float32)
    y = cv_labels[i].astype(np.int)
    
    
    lst = list(model2.parameters())
    for i in range(1000):
        
        h1 = x[i].dot(lst[0].data.numpy().transpose()) + lst[1].data.numpy()
        h1_relu = np.maximum(0.0, h1)
        h2 = h1_relu.dot(lst[2].data.numpy().transpose()) + lst[3].data.numpy()
        h2_relu = np.maximum(0.0, h2)
        y_pred = h2_relu.dot(lst[4].data.numpy().transpose()) + lst[5].data.numpy()
        res = np.argmax(y_pred)
        y_hat[i] = res
    acc = accuracy(y,y_hat)
    return acc


train_accs, val_accs = [], []
batch_size = 100
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))

import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()


HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS, H = 26, 26, 5, 5000, 300
learning_rate = 0.0001

class FeedForwardNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(FeedForwardNN, self).__init__()
        #self.Linear = torch.nn.Linear(D_in, D_out)
        #self.hidden_layer_count = layers
        self.Linear1 = torch.nn.Linear(D_in, H)
        self.drop = torch.nn.Dropout(p=0.5, inplace=False)
        self.middleLinear = torch.nn.Linear(H, H)
        self.Linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h = self.Linear1(x)
        h_relu = F.relu(h, inplace=False)
        #x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
        h_middle = self.middleLinear(self.drop(h_relu))
        h_middle_relu = F.relu(h_middle, inplace = False)
        y_pred = self.Linear2(h_middle)
        return y_pred
        
    
model3 = FeedForwardNN(HEIGHT * WIDTH, NUM_CLASSES)
optimizer = torch.optim.Adam(model3.parameters(), lr= learning_rate)
def train(batch_size):
    model3.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model3(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]

import random
def approx_train_accuracy():
    model3.eval()
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = train_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    lst = list(model3.parameters())
        
    for i in range(1000):
        res = model3(x[i])
        y_hat[i] = np.argmax(res.data.numpy())
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    model3.eval()
    y_hat = np.empty(1000)

    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(val_seqs[i].astype(np.float32)))
    y = cv_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    
    lst = list(model3.parameters())
    for i in range(1000):
        
        res = model3(x[i])
        y_hat[i] = np.argmax(res.data.numpy())
    acc = accuracy(y,y_hat)
    return acc

train_accs, val_accs = [], []
batch_size = 100
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))

import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()

HEIGHT, WIDTH, NUM_CLASSES, NUM_OPT_STEPS, H = 26, 26, 5, 5000, 300
learning_rate = 0.0001

class FeedForwardNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(FeedForwardNN, self).__init__()
        self.Linear1 = torch.nn.Linear(D_in, H)
        self.middleLinear = torch.nn.Linear(H, H)
        self.Linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        h = self.Linear1(x)
        h_relu = F.relu(h, inplace=False)
        h_drop = F.dropout(h_relu, training = self.training)
        h_middle = self.middleLinear(h_drop)
        h_middle_relu = F.relu(h_middle, inplace = False)
        y_pred = self.Linear2(h_middle_relu)
        return y_pred
        
    
model3 = FeedForwardNN(HEIGHT * WIDTH, NUM_CLASSES)
optimizer = torch.optim.Adam(model3.parameters(), lr= learning_rate)
def train(batch_size):
    model3.train()
    
    i = np.random.choice(train_seqs.shape[0], size = batch_size, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = Variable(torch.from_numpy(train_labels[i].astype(np.int)))
    
    optimizer.zero_grad()
    y_hat = model3(x)
    loss = F.multi_margin_loss(y_hat, y)
    loss.backward()
    optimizer.step()
    
    return loss.data[0]

def accuracy(y, y_hat):
    count = 0
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            count += 1
    return count/y.shape[0]

import random
def approx_train_accuracy():
    model3.eval()
    i = np.random.choice(train_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(train_seqs[i].astype(np.float32)))
    y = train_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    lst = list(model3.parameters())
        
    for i in range(1000):
        res = model3(x[i])
        y_hat[i] = np.argmax(res.data.numpy())
    acc = accuracy(y,y_hat)
    return acc

def val_accuracy():
    model3.eval()
    y_hat = np.empty(1000)

    i = np.random.choice(val_seqs.shape[0], size = 1000, replace=False)
    x = Variable(torch.from_numpy(val_seqs[i].astype(np.float32)))
    y = cv_labels[i].astype(np.int)
    y_hat = np.empty(1000)
    
    
    lst = list(model3.parameters())
    for i in range(1000):
        
        res = model3(x[i])
        y_hat[i] = np.argmax(res.data.numpy())
    acc = accuracy(y,y_hat)
    return acc


train_accs, val_accs = [], []
batch_size = 200
for i in range(5000):
    l = train(batch_size)
    if i % 100 == 0:
        #model3.eval()
        train_accs.append(approx_train_accuracy())
        val_accs.append(val_accuracy())
        print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))

import matplotlib.pyplot as plt


t = np.arange(0,len(train_accs),1)

s = train_accs
k = val_accs
print("max_train accuracy: ", max(train_accs))
print("max_val accuracy: ", max(val_accs))
plt.figure(figsize=(8,8), dpi = 80)
plt.plot(t, s, t, k)

plt.xlabel('number of iteration')
plt.ylabel('accuracy')
plt.title('Training/validation accuracy')
plt.grid(True)
plt.show()

