import numpy as np
import tensorflow as tf
import gym
from tensorflow.contrib import slim

class Memory(object):
    def __init__(self, dimension=10, size=10000, ):
        self.memory = np.empty((dimension,size), dtype=np.float32)
        self.size = size
        self.index = 0 # keeps track of current size
        self.full = False
    def add(self, memory):
        self.memory[:, self.index] = memory
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
        return self.memory[:, idx]

#class Layer():
#    def __init__(self):
#        pass
#
#def DenseLayer(Layer):
#    def __init__(self, shape):
#        self.shape = shape
#        self.W = tf.get_variable("W%d"%DenseLayer.cnt, shape = shape, initializer = tf.contrib.layers.xavier_initializer())
#    def apply(self, x):
#        return tf.matmul(x, self.W)
#    def copyTo(self, l, tau):
#        if tau == 1.0:
#            # hard update
#            return [l.W.assign(self.W)]
#        else:
#            return [l.W.assign(tau * self.W.value() + (1-tau) * l.W.value())]

unique_cnt = 0
def unique_name(suffix):
    global unique_cnt
    unique_cnt += 1
    return suffix + '_' + str(unique_cnt)

class QNet(object):
    def __init__(self, scope_name='net'):
        self.scope = scope_name

        with tf.variable_scope(scope_name) as scope:
            self.inputs = tf.placeholder(shape=(4,32), dtype=tf.float32)

            self.W1 = tf.get_variable(unique_name('W'), shape = (64,4), initializer = tf.contrib.layers.xavier_initializer())
            hidden1 = tf.nn.tanh(tf.matmul(self.W, self.inputs))
            self.W2 = tf.get_variable(unique_name('W'), shape = (2,4), initializer = tf.contrib.layers.xavier_initializer())
            hidden2 = tf.nn.tanh(tf.matmul(self.W2, hidden1))

            hidden = slim.fully_connected(self.inputs, 64,activation_fn = tf.nn.tanh, biases_initializer=None)
            self._Q = slim.fully_connected(hidden, 2, activation_fn = None, biases_initializer = None)
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
        ops.append(dv.assign(lerp(sv,dv,tau)))
    return ops

if __name__ == "__main__":
    ## Load Parameters
    gamma = .99 #Discount factor.
    num_episodes = 20000 #Total number of episodes to train network for.
    tau = 0.001 #Amount to update target network at each step.
    batch_size = 32 #Size of training batch

    eps_start = 1 #Starting chance of random action
    eps_end = 0.1 #Final chance of random action
    annealing_steps = 200000 #How many steps of training to reduce startE to endE.
    eps_delta = (eps_end - eps_start) / annealing_steps

    pre_train_steps = 50000 #Number of steps us

    ## Start Environment
    env = gym.make('CartPole-v0')
    state_size = reduce(lambda x,y:x*y, env.observation_space.shape)
    action_size = env.action_space.n

    tf.reset_default_graph()
    session = tf.Session()

    # initialize memory
    mem_size = 2*state_size + 1 + 1 + 1 # 1=action, 1=reward, 1 = done
    memory = Memory(mem_size, 10000) # default size = 10000

    # get networks
    net = QNet('net')
    target_net = QNet('target')
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
            if np.random.rand(1) < eps or step < pre_train_steps:
                a = env.action_space.sample()
            else:
                a = session.run(net.predict, feed_dict={net.inputs:[s]})
                a = a[0]

            s1,r,d,_ = env.step(a)
            entry = np.hstack((s0,a,r,s1,d))
            memory.add(entry) # column vector
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
                    _s0 = input_batch[0:4, :]
                    _a = input_batch[4,:]
                    _r = input_batch[5,:]
                    _s1 = input_batch[6:10,:]
                    _d = input_batch[10,:]

                    # s1 = (4,32), --> (x,32)
                    a_s1 = session.run(net.predict, feed_dict={net.inputs : _s1})
                    q_s1 = session.run(target_net._Q, feed_dict={target_net.inputs : _s1})
                    print q_s1.shape
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
            print "Mean Reward: %f; Step : %d, Epsilon: %f" % (r_mean, step, eps)
