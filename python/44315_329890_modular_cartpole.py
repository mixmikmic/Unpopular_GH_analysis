get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

import numpy as np
import gym
import tensorflow as tf

class Actor():
    def __init__(self, n_obs, h, n_actions):
        self.n_obs = n_obs                  # dimensionality of observations
        self.h = h                          # number of hidden layer neurons
        self.n_actions = n_actions          # number of available actions
        
        self.model = model = {}
        with tf.variable_scope('actor_l1',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
            model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
        with tf.variable_scope('actor_l2',reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
            model['W2'] = tf.get_variable("W2", [h,n_actions], initializer=xavier_l2)
            
    def policy_forward(self, x): #x ~ [1,D]
        h = tf.matmul(x, self.model['W1'])
        h = tf.nn.relu(h)
        logp = tf.matmul(h, self.model['W2'])
        p = tf.nn.softmax(logp)
        return p

class Agent():
    def __init__(self):
        self.gamma = .9             # discount factor for reward
        self.xs, self.rs, self.ys = [],[],[]
        
        self.actor_lr = 1e-2        # learning rate for policy
        self.decay = 0.9
        self.n_obs = n_obs = 4              # dimensionality of observations
        self.n_actions = n_actions = 2          # number of available actions
        
        # make actor part of brain
        self.actor = Actor(n_obs=self.n_obs, h=128, n_actions=self.n_actions)
        
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
        self.optimizer = tf.train.RMSPropOptimizer(self.actor_lr, decay=self.decay)
        self.grads = self.optimizer.compute_gradients(self.loss,                                     var_list=tf.trainable_variables(), grad_loss=self.discounted_r)
        self.train_op = self.optimizer.apply_gradients(self.grads)

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
    
    def act(self, x):
        feed = {self.x: x}
        aprob = self.sess.run(self.aprob, feed) ; aprob = aprob[0,:]
        action = np.random.choice(self.n_actions, p=aprob)
        
        label = np.zeros_like(aprob) ; label[action] = 1
        self.xs.append(x)
        self.ys.append(label)
        
        return action
    
    def learn(self):
        epx = np.vstack(self.xs)
        epr = np.vstack(self.rs)
        epy = np.vstack(self.ys)
        self.xs, self.rs, self.ys = [],[],[] # reset game history
        
        feed = {self.x: epx, self.r: epr, self.y: epy}
        _ = self.sess.run(self.train_op,feed) # parameter update
        
    @staticmethod
    def discount_rewards(r, gamma):
        discount_f = lambda a, v: a*gamma + v;
        r_reverse = tf.scan(discount_f, tf.reverse(r,[True, False]))
        discounted_r = tf.reverse(r_reverse,[True, False])
        return discounted_r

def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()

agent = Agent()
env = gym.make("CartPole-v0")
observation = env.reset()
running_reward = 10 # usually starts around 10 for cartpole
reward_sum = 0
episode_number = 0
total_steps = 500

fig,ax = plt.subplots(1,1)
ax.set_xlabel('X') ; ax.set_ylabel('Y')
ax.set_xlim(0,total_steps) ; ax.set_ylim(0,200)
pxs, pys = [], []

print 'episode {}: starting up...'.format(episode_number)
while episode_number <= total_steps and running_reward < 225:
#     if episode_number%25==0: env.render()

    # stochastically sample a policy from the network
    x = observation
    action = agent.act(np.reshape(x, (1,-1)))

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    agent.rs.append(reward)
    reward_sum += reward
    
    if done:
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        agent.learn()

        # visualization
        pxs.append(episode_number)
        pys.append(running_reward)
        if episode_number % 25 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            plt_dynamic(pxs, pys, ax)
        
        # lame stuff
        episode_number += 1 # the Next Episode
        observation = env.reset() # reset env
        reward_sum = 0
        
plt_dynamic(pxs, pys, ax)
if running_reward > 225:
    print "ep: {}: SOLVED! (running reward hit {} which is greater than 200)".format(
        episode_number, running_reward)



