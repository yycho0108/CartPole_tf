#!/usr/bin/python
import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

class Memory(object):
    def __init__(self, dimension=10, size=10000):
        self.memory = np.empty((size,dimension), dtype=np.float32)
        self.size = size
        self.index = 0 # keeps track of current size
        self.full = False
    def add(self, memory):
        self.memory[self.index,:] = memory
        self.index += 1
        if self.index >= self.size:
            self.index = 0
            self.full = True
    def sample(self, n):
        # data stored in columns
        # each column is one entry
        if self.full:
            idx = np.random.randint(self.size, size=n)
        else:
            idx = np.random.randint(self.index, size=n)
        return self.memory[idx,:]

class Layer(object):
    cnt = 0
    def __init__(self):
        pass
    def apply(self,x):
        pass
    @staticmethod
    def name(suffix):
        Layer.cnt += 1
        return str(suffix) + '_' + str(Layer.cnt)

class DenseLayer(Layer):
    def __init__(self, shape):
        super(DenseLayer,self).__init__()
        self.W = tf.get_variable(Layer.name('W'), shape = shape, initializer = tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable(Layer.name('b'), initializer = tf.zeros((shape[-1],)))

    def apply(self, x):
        return tf.matmul(x, self.W) + self.b


class ActivationLayer(Layer):
    def __init__(self,t):
        super(ActivationLayer,self).__init__()
        self.type = t

    def apply(self,x):
        if self.type == 'relu':
            return tf.nn.relu(x)
        elif self.type == 'softmax':
            return tf.nn.softmax(x)
        elif self.type == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif self.type == 'tanh':
            return tf.nn.tanh(x)
        else:
            #elif self.type == 'linear':
            return x

unique_cnt = 0
def unique_name(suffix=''):
    global unique_cnt
    unique_cnt += 1
    return suffix + '_' + str(unique_cnt)

class QNet(object):
    def __init__(self, in_dim):
        self.scope = tf.get_variable_scope().name
        self.inputs = tf.placeholder(shape=(None,in_dim), dtype=tf.float32)
        self.L = []

    def append(self, l):
        self.L.append(l)

    def setup(self):
        X = self.inputs
        for l in self.L:
            X = l.apply(X)
        self._Q = X 

        self.predict = tf.argmax(self._Q, 1)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_one_hot = tf.one_hot(self.actions, 2, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.mul(self._Q, self.actions_one_hot), reduction_indices=1)
        self.Qn = tf.placeholder(shape=[None], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.Qn - self.Q))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)


        self.update = trainer.minimize(loss)


def lerp(a,b,w):
    return w*a + (1.-w)*b

def get_copy_ops(src, dst, tau):
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src.scope)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dst.scope)
    ops = []
    for sv, dv in zip(src_vars, dst_vars):
        print sv.name
        print dv.name
        ops.append(dv.assign(lerp(sv,dv,tau)))
    print ops
    return ops

if __name__ == "__main__":
    ## Load Parameters
    gamma = .99 #Discount factor.
    num_episodes = 20000 #Total number of episodes to train network for.
    tau = 0.001 #Amount to update target network at each step.
    batch_size = 32 #Size of training batch

    eps_start = 1 #Starting chance of random action
    eps_end = 0.05 #Final chance of random action
    annealing_steps = 200000 #How many steps of training to reduce startE to endE.
    eps_delta = (eps_end - eps_start) / annealing_steps

    pre_train_steps = 50000 #Number of steps us

    ## Start Environment
    state_size = reduce(lambda x,y:x*y, env.observation_space.shape)
    action_size = env.action_space.n

    tf.reset_default_graph()
    session = tf.Session()

    # initialize memory
    mem_size = 2*state_size + 1 + 1 + 1 # 1=action, 1=reward, 1 = done
    memory = Memory(mem_size, 10000) # default size = 10000

    # get networks
    with tf.variable_scope('net') as scope:
        net = QNet(4)
        net.append(DenseLayer((4,64)))
        net.append(ActivationLayer('tanh'))
        net.append(DenseLayer((64,2)))
        net.setup()
    with tf.variable_scope('target') as scope:
        target_net = QNet(4)
        target_net.append(DenseLayer((4,64)))
        target_net.append(ActivationLayer('tanh'))
        target_net.append(DenseLayer((64,2)))
        target_net.setup()

    copy_ops = get_copy_ops(net, target_net, tau)

    # get this started...
    session.run(tf.initialize_all_variables())
    session.run(copy_ops)

    eps = eps_start
    step = 0

    rewards = []

    for i in range(num_episodes):
        s0 = env.reset()
        net_reward = 0
        d = False
        while not d:
            #if i > num_episodes/2:
            #    env.render()
            if np.random.rand(1) < eps or step < pre_train_steps:
                a = env.action_space.sample()
            else:
                a = session.run(net.predict, feed_dict={net.inputs:[s0]})
                a = a[0]

            s1,r,d,_ = env.step(a)
            entry = np.hstack((s0,a,r,s1,d)) # row vec
            memory.add(entry)
            #s0,a,r,s1,d
            # start index
            #s0 [0]
            #a [4]
            #r [5]
            #s1 [6]
            #d [10]

            if step > pre_train_steps:
                if eps > eps_end:
                    eps += eps_delta
                if step % 5 == 0:
                    input_batch = memory.sample(batch_size)
                    _s0 = input_batch[:, 0:4]
                    _a = input_batch[:, 4]
                    _r = input_batch[:, 5]
                    _s1 = input_batch[:, 6:10]
                    _d = input_batch[:, 10]

                    # s1 = (4,32), --> (x,32)
                    a_s1 = session.run(net.predict, feed_dict={net.inputs : _s1})
                    q_s1 = session.run(target_net._Q, feed_dict={target_net.inputs : _s1})
                    q = q_s1[range(batch_size), a_s1]
                    target_q = _r + gamma * q * (1 - _d)
                    _ = session.run(net.update, feed_dict = {net.inputs : _s0, net.Qn:target_q, net.actions:_a})
                    session.run(copy_ops)
            net_reward += r
            s0 = s1
            step += 1
        rewards.append(net_reward)

        if i % 100 == 0 and i > 0:
            r_mean = np.mean(rewards[-100:])
            print "Epoch : %d, Mean Reward: %f; Step : %d, Epsilon: %f" % (i, r_mean, step, eps)
