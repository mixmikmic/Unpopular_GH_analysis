get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import time
import heapq

import numpy as np
import gym
import tensorflow as tf

# class ActorConv():
#     def __init__(self, n_obs, h, n_actions):
#         self.n_obs = n_obs                  # dimensionality of observations
#         self.h = h                          # number of hidden layer neurons
#         self.n_actions = n_actions          # number of available actions
        
#         self.model = model = {}
#         with tf.variable_scope('actor',reuse=False):
#             # convolutional layer 1
#             self.model['Wc1'] = tf.Variable(tf.truncated_normal([4, 4, 2, 8], stddev=0.1))
#             self.model['bc1'] = tf.Variable(tf.constant(0.1, shape=[8]))

#             # convolutional layer 2
#             self.model['Wc2'] = tf.Variable(tf.truncated_normal([4, 4, 8, 8], stddev=0.1))
#             self.model['bc2'] = tf.Variable(tf.constant(0.1, shape=[8]))

#             # fully connected 1
#             self.model['W3'] = tf.Variable(tf.truncated_normal([14*10*8, self.h], stddev=0.1))
#             self.model['b3'] = tf.Variable(tf.constant(0.1, shape=[self.h]))

#             # fully connected 2
#             self.model['W4'] = tf.Variable(tf.truncated_normal([self.h, n_actions], stddev=0.1))
#             self.model['b4'] = tf.Variable(tf.constant(0.1, shape=[n_actions]))
            
#     def policy_forward(self, x):
#         x_image = tf.reshape(x, [-1, 105, 80, 2])
                                      
#         zc1 = tf.nn.conv2d(x_image, self.model['Wc1'], strides=[1, 1, 1, 1], padding='SAME') + self.model['bc1']
#         hc1 = tf.nn.relu(zc1)
#         hc1 = tf.nn.max_pool(hc1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
#         zc2 = tf.nn.conv2d(hc1, self.model['Wc2'], strides=[1, 1, 1, 1], padding='SAME') + self.model['bc2']
#         hc2 = tf.nn.relu(zc2)
#         hc2 = tf.nn.max_pool(hc2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#         print hc2.get_shape()
        
#         hc2_flat = tf.reshape(hc2, [-1, 14*10*8])
#         h3 = tf.nn.relu(tf.matmul(hc2_flat, self.model['W3']) + self.model['b3'])
#         h3 = tf.nn.dropout(h3, 0.9)
        
#         h4 = tf.matmul(h3, self.model['W4']) + self.model['b4']
#         return tf.nn.softmax(h4)

class Actor():
    def __init__(self, n_obs, h, n_actions):
        self.n_obs = n_obs                  # dimensionality of observations
        self.h = h                          # number of hidden layer neurons
        self.n_actions = n_actions          # number of available actions
        
        self.model = model = {}
        with tf.variable_scope('actor_l1',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
            model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
            model['b1'] = tf.get_variable("b1", [1, h], initializer=xavier_l1)
        with tf.variable_scope('actor_l2',reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
            model['W2'] = tf.get_variable("W2", [h,n_actions], initializer=xavier_l2)
            model['b2'] = tf.get_variable("b1", [1, n_actions], initializer=xavier_l2)
            
    def policy_forward(self, x): #x ~ [1,D]
        h = tf.matmul(x, self.model['W1']) + self.model['b1']
        h = tf.nn.relu(h)
        logp = tf.matmul(h, self.model['W2']) + self.model['b2']
        p = tf.nn.softmax(logp)
        return p

class Agent():
    def __init__(self, n_obs, n_actions, gamma=0.99, actor_lr = 1e-4, decay=0.95, epsilon = 0.1):
        self.gamma = gamma            # discount factor for reward
        self.epsilon = epsilon
        self.global_step = 0
        self.replay_max = 32 ; self.replay = []
        self.xs, self.rs, self.ys = [],[],[]
        
        self.actor_lr = actor_lr               # learning rate for policy
        self.decay = decay
        self.n_obs = n_obs                     # dimensionality of observations
        self.n_actions = n_actions             # number of available actions
        self.save_path ='models/pong.ckpt'
        
        # make actor part of brain
        self.actor = Actor(n_obs=self.n_obs, h=200, n_actions=self.n_actions)
        
        #placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="y")
        self.r = tf.placeholder(dtype=tf.float32, shape=[None,1], name="r")
        
        #gradient processing (PG magic)
        self.discounted_r = self.discount_rewards(self.r, self.gamma)
        mean, variance= tf.nn.moments(self.discounted_r, [0], shift=None, name="reward_moments")
        self.discounted_r -= mean
        self.discounted_r /= tf.sqrt(variance + 1e-6)
        
        # initialize tf graph
        self.aprob = self.actor.policy_forward(self.x)
        self.loss = tf.nn.l2_loss(self.y-self.aprob)
        self.optimizer = tf.train.RMSPropOptimizer(self.actor_lr, decay=self.decay, momentum=0.25)
        self.grads = self.optimizer.compute_gradients(self.loss,                                     var_list=tf.trainable_variables(), grad_loss=self.discounted_r)
        self.train_op = self.optimizer.apply_gradients(self.grads)

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(tf.all_variables())
    
    def act(self, x):
        aprob = self.sess.run(self.aprob, {self.x: x})
        aprob = aprob[0,:]
        if np.random.rand() > 0.9998: print "\taprob is: ", aprob
        action = np.random.choice(self.n_actions,p=aprob) if np.random.rand() > self.epsilon else np.random.randint(self.n_actions)
        
        label = np.zeros_like(aprob) ; label[action] = 1
        self.xs.append(x)
        self.ys.append(label)
        
        return action
    
    def learn(self, n):
        epx = np.vstack(self.xs)
        epr = np.vstack(self.rs)
        epy = np.vstack(self.ys)
        self.append_replay([epx, epr, epy])
        self.xs, self.rs, self.ys = [],[],[] # reset game history
        
#         self.apply_rewards(epx,epr,epy)
        self.learn_replay(n)
        self.global_step += 1
        
    def apply_rewards(self, epx, epr, epy):
        feed = {self.x: epx, self.r: epr, self.y: epy}
        _ = self.sess.run(self.train_op,feed) # parameter update
        self.global_step += 1
        
    def append_replay(self, ep):
        self.replay.append(ep)
        if len(self.replay) is self.replay_max + 1:
            self.replay = self.replay[1:]
            
    def learn_replay(self, n):
        assert n <= self.replay_max, "requested number of entries exceeds epmax"
        if len(self.replay) < self.replay_max: print "\t\tqueue too small" ; return
        ix = np.random.permutation(self.replay_max)[:n]
        epx = np.vstack([ self.replay[i][0] for i in ix])
        epr = np.vstack([ self.replay[i][1] for i in ix])
        epy = np.vstack([ self.replay[i][2] for i in ix])
        self.apply_rewards(epx, epr, epy)
        
    @staticmethod
    def discount_rewards(r, gamma):
        discount_f = lambda a, v: a*gamma + v;
        r_reverse = tf.scan(discount_f, tf.reverse(r,[True, False]))
        discounted_r = tf.reverse(r_reverse,[True, False])
        return discounted_r

# downsampling
# def prepro(o):
#     rgb = o
#     gray = 0.3*rgb[:,:,0:1] + 0.4*rgb[:,:,1:2] + 0.3*rgb[:,:,2:3]
#     gray = gray[::2,::2,:]
#     gray -= np.mean(gray) ; gray /= 100
#     return gray.astype(np.float)
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()

n_obs = 80*80 #2*105*80   # dimensionality of observations
n_actions = 3
agent = Agent(n_obs, n_actions, gamma = 0.99, actor_lr=1e-3, decay=0.95, epsilon = 0.1)

save_path = 'models/model.ckpt'
saver = tf.train.Saver(tf.all_variables())

saver = tf.train.Saver(tf.all_variables())
load_was_success = True # yes, I'm being optimistic
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(agent.sess, load_path)
except:
    print "no saved model to load. starting new session"
    load_was_success = False
else:
    print "loaded model: {}".format(load_path)
    saver = tf.train.Saver(tf.all_variables())
    agent.global_step = int(load_path.split('-')[-1])

env = gym.make("Pong-v0")
observation = env.reset()
cur_x = None
prev_x = None
running_reward = -20.48 # usually starts around 10 for cartpole
reward_sum = 0
episode_number = 0

total_parameters = 0 ; print "Model overview:"
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    print '\tvariable "{}" has {} parameters'         .format(variable.name, variable_parameters)
    total_parameters += variable_parameters
print "Total of {} parameters".format(total_parameters)

fig,ax = plt.subplots(1,1)
ax.set_xlabel('X') ; ax.set_ylabel('Y')
ax.set_xlim(0,500) ; ax.set_ylim(-21,-19)
pxs, pys = [], []

print 'episode {}: starting up...'.format(episode_number)
start = time.time()
while True:
#     if episode_number%25==0: env.render()

    # preprocess the observation, set input to network to be difference image
#     prev_x = cur_x if cur_x is not None else np.zeros((105,80,1))
#     cur_x = prepro(observation)
#     x = np.concatenate((cur_x, prev_x),axis=-1).ravel()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
    prev_x = cur_x

    # stochastically sample a policy from the network
    action = agent.act(np.reshape(x, (1,-1)))

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action + 1)
    agent.rs.append(reward)
    reward_sum += reward
    
    if done:
        eptime = time.time()
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        agent.learn(16)

        # visualization
        pxs.append(episode_number)
        pys.append(running_reward)
        if episode_number % 10 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            plt_dynamic(pxs, pys, ax)
        else:
            print '\tep: {}, reward: {}'.format(episode_number, reward_sum)
            
#         if episode_number % 50 == 0:
#             saver.save(agent.sess, save_path, global_step=agent.global_step)
#             print "SAVED MODEL #{}".format(agent.global_step)

        stop = time.time()
        if episode_number % 10 == 0:
            print "\t\teptime: {}".format(eptime - start)
            print "\t\tlearntime: {}".format(stop - eptime)
        start = stop
        
        # lame stuff
        cur_x = None
        episode_number += 1 # the Next Episode
        observation = env.reset() # reset env
        reward_sum = 0

saver.save(agent.sess, save_path, global_step=agent.global_step)

def prepro(o):
    rgb = o
    gray = 0.3*rgb[:,:,0:1] + 0.4*rgb[:,:,1:2] + 0.3*rgb[:,:,2:3]
    gray = gray[::2,::2,:]
    gray -= np.mean(gray) ; gray /= 100
    return gray.astype(np.float)

print np.reshape(x, (1,-1)).shape
print agent.x.get_shape()
agent.act(np.reshape(x, (1,-1)))

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = prepro(observation)
observation, reward, done, info = env.step(2)
cur_x = prepro(observation)

get_ipython().magic('matplotlib inline')
print cur_x.shape
plt.imshow(cur_x[:,:,0])

x = np.concatenate((cur_x, prev_x),axis=-1).ravel()
print x.shape

p = np.reshape(x,(-1,105,80,2))
plt.imshow(p[0,:,:,1])

plt.imshow(p[0,:,:,0])



