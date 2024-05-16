import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x_data = np.random.rand(100).astype(np.float32)

y_data = 3 * x_data + 2
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0,scale=0.1))(y_data)

zipped = zip(x_data,y_data)

print(zipped)

w = tf.Variable(1.0)
b = tf.Variable(0.2)
y = w * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_data = []
for step in range(1000):
    evals = sess.run([train, w, b])[1:]
    if step % 50 == 0:
        print(step,evals)
        train_data.append(evals)

converter = plt.colors 
cr,cg, cb = (1.0,1.0,0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0 : cb = 1.0
    if cg < 0.0 : cg = 0.0
    [w,b] = f
    
    f_y = np.vectorize(lambda x: w*x + b) (x_data)
    line = plt.plot(x_data,f_y)
    plt.setp(line,color=(cr,cg,cb))
    
plt.plot(x_data,y_data,'ro')

green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()

