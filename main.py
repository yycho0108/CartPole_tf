#!/usr/bin/python
import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()

import numpy as np
import tensorflow as tf

from memory import Memory
from qnet import *
from layers import *


## Load Parameters
gamma = .99 #Discount factor.
num_episodes = 2000 #Total number of episodes to train network for.
test_episodes = 2000 #Total number of episodes to train network for.
tau = 0.001 #Amount to update target network at each step.
batch_size = 32 #Size of training batch

eps_start = 1 #Starting chance of random action
eps_end = 0.05 #Final chance of random action
annealing_steps = 200000 #How many steps of training to reduce startE to endE.
eps_delta = (eps_end - eps_start) / annealing_steps

pre_train_steps = 50000 #Number of steps us

## Initialize Tensorflow
tf.reset_default_graph()
session = tf.Session()

def train(net, target_net, memory, episodes):
    eps = eps_start
    step = 0

    rewards = []

    for i in range(episodes):
        s0 = env.reset()
        net_reward = 0
        d = False
        while not d and net_reward < 999:
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
    return rewards

def test(net, episodes):
    rewards = []

    # test
    for i in range(episodes):
        s = env.reset()
        d = False
        net_reward = 0
        while not d and net_reward < 999:
            env.render()
            a = session.run(net.predict, feed_dict={net.inputs:[s]})
            a = a[0]
            s,r,d,_ = env.step(a)
            net_reward += r
        rewards.append(net_reward)
    return rewards

def setup():
    ## Start Environment
    state_size = reduce(lambda x,y:x*y, env.observation_space.shape)

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
    return net, target_net, memory

def main():
    net, target_net, memory = setup()
    train_rewards = train(net, target_net, memory, num_episodes)
    test_rewards = test(net, test_episodes)

    np.savetxt('train.csv', train_rewards, delimiter=',', fmt='%f')
    np.savetxt('test.csv', test_rewards, delimiter=',', fmt='%f')

if __name__ == "__main__":
    main()
    
