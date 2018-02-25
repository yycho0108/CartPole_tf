#!/usr/bin/python
"""
main_rnn.py

@Author : Yoonyoung Cho
@Date : 02/25/2018

Description : 
    Modification of main.py to run CartPole with recurrent neural networks (LSTMs)
    with DRQN-like architecture, following a [tutorial](https://github.com/awjuliani/DeepRL-Agents/blob/master/Deep-Recurrent-Q-Network.ipynb).
    Note that the CartPole model does not receive velocity-related data as input, meaning that it should learn to model the system motion.

Notes : 
    Still modifying train()
    None of the code below that will work.
"""

import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()

import sys
import numpy as np
import tensorflow as tf

from memory import TraceMemory
from drqn import DRQN

## Load Parameters
N_H = 64
U_FREQ = 16

gamma = .99 #Discount factor.
num_episodes = 10000 #Total number of episodes to train network for.
test_episodes = 200 #Total number of episodes to train network for.
tau = 0.001 #Amount to update target network at each step.
batch_size = 32 #Size of training batch

eps_start = 1 #Starting chance of random action
eps_end = 0.05 #Final chance of random action
annealing_steps = 200000 #How many steps of training to reduce startE to endE.
eps_delta = (eps_end - eps_start) / annealing_steps

pre_train_steps = 50000 #Number of steps us

## Initialize Tensorflow
tf.reset_default_graph()
sess= tf.Session()

copy_ops = None

def run(sess, tensors, ins_t, ins_v, outs):
    return sess.run([tensors[e] for e in outs],
            feed_dict = {tensors[t]:v for t,v in zip(ins_t, ins_v)})

def proc(x):
    # remove velocity information!
    return [x[0], x[2]]

def train(net, target_net, memory, episodes):
    eps = eps_start
    step = 0
    rewards = []
    max_step = 999 # TODO : configure this

    def train_once(eps, step):
        s0 = proc(env.reset())

        # initialize episode
        net_reward = 0
        d = False
        c = np.zeros([1, N_H])
        h = np.zeros([1, N_H])
        entry = []

        for j in range(max_step):

            if np.random.rand(1) < eps or step < pre_train_steps:
                c1, h1 = run(sess, net._tensors, ['x_in', 'c_in', 'h_in', 'n_b', 'n_t'],
                        [s0, c, h, 1, 1], ['c_out', 'h_out'])
                a = env.action_space.sample()
            else:
                a, c1, h1 = run(sess, net._tensors, ['x_in', 'c_in', 'h_in', 'n_b', 'n_t'],
                        [s0, c, h, 1, 1], ['a_y', 'c_out', 'h_out'])
                a = a[0]

            s1,r,d,_ = env.step(a)
            s1 = proc(s1)

            entry.append(s0 + [a,r] + s1 + [d]) #[0:2, 2, 3, 4:6, 6]

            if step > pre_train_steps:
                if eps > eps_end:
                    eps += eps_delta
                if step % U_FREQ == 0:
                    # TODO: update(), main --> target
                    input_batch = memory.sample(batch_size)
                    _s0 = input_batch[:, 0:4]
                    _a = input_batch[:, 4]
                    _r = input_batch[:, 5]
                    _s1 = input_batch[:, 6:10]
                    _d = input_batch[:, 10]

                    # s1 = (4,32), --> (x,32)
                    a_s1 = sess.run(net.predict, feed_dict={net.inputs : _s1})
                    q_s1 = sess.run(target_net._Q, feed_dict={target_net.inputs : _s1})
                    q = q_s1[range(batch_size), a_s1]
                    target_q = _r + gamma * q * (1 - _d)
                    _ = sess.run(net.update, feed_dict = {net.inputs : _s0, net.Qn:target_q, net.actions:_a})
                    sess.run(copy_ops)
            net_reward += r
            s0 = s1
            step += 1

            if d:
                break
        return net_reward, eps, step

    r_mean = 0
    i = 0
    
    while i < episodes: 
        net_reward, eps, step = train_once(eps, step)
        rewards.append(net_reward)

        if i % 100 == 0 and i > 0:
            r_mean = np.mean(rewards[-100:])
            print "Epoch : %d, Mean Reward: %f; Step : %d, Epsilon: %f" % (i, r_mean, step, eps)
        i += 1

    if raw_input('Train Until Convergence?\n').lower() == 'y':
        while r_mean < 999:
            net_reward, eps, step = train_once(eps, step)
            rewards.append(net_reward)

            if i % 100 == 0 and i > 0:
                r_mean = np.mean(rewards[-100:])
                print "Epoch : %d, Mean Reward: %f; Step : %d, Epsilon: %f" % (i, r_mean, step, eps)
            i += 1


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
            a = sess.run(net.predict, feed_dict={net.inputs:[s]})
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

    return net, target_net, memory

def main():
    global copy_ops

    net, target_net, memory = setup()

    copy_ops = get_copy_ops(net, target_net, tau)
    # get this started...
    sess.run(tf.initialize_all_variables())
    sess.run(copy_ops)
    saver = tf.train.Saver()

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'load':
        saver.restore(sess, '/tmp/model.ckpt')
        print '[loaded]'
    else:
        train_rewards = train(net, target_net, memory, num_episodes)
        np.savetxt('train.csv', train_rewards, delimiter=',', fmt='%f')
        save_path = saver.save(sess, '/tmp/model.ckpt')
        print("Model saved in file: %s" % save_path) 

    test_rewards = test(net, test_episodes)
    np.savetxt('test.csv', test_rewards, delimiter=',', fmt='%f')

if __name__ == "__main__":
    main()
    
