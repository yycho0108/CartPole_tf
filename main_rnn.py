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

from utils import *

import gym
env = gym.make('CartPole-v0')
env.reset()
#env.render()

import os
import sys
import numpy as np
import tensorflow as tf

from memory import TraceMemory
from drqn import DRQN

## Network/Meta Parameters
N_X = 4 # size of input
N_A = 2 # size of action
N_H = 64 # number of hidden units
U_FREQ = 4 # update frequency
N_LOG = 16
N_BATCH = 32 # size of training batch
N_TRACE = 8
LEARNING_RATE = .0001

## Q-Learning Parameters
GAMMA = .99 #Discount factor.
N_EPOCH = np.inf #20000 #Total number of episodes to train network for.
N_TEST = 200 #Total number of episodes to train network for.
TAU = 0.001 #Amount to update target network at each step.

# Annealing Parameters
EPS_INIT  = 1.00 #Starting chance of random action
EPS_MIN  = 0.01 #Final chance of random action
N_ANNEAL = 400000 #How many steps of training to reduce startE to endE.
#EPS_DECAY = np.log(EPS_MIN)/N_ANNEAL
EPS_DECAY = 0.995

N_PRE = 50000 #Number of steps, pre-train
N_MEM = 10000

## Initialize Tensorflow
tf.reset_default_graph()
sess= tf.Session()

dirs = directory_setup('drqn')

def run(sess, tensors, ins_t, ins_v, outs):
    return sess.run([tensors[e] for e in outs],
            feed_dict = {tensors[t]:v for t,v in zip(ins_t, ins_v)})

def proc(x):
    # remove velocity information
    # (x,v,t,w) -> (x,t)
    #return [x[0], x[2]]
    # decompose t -> sin(t), cos(t)
    return list(x)

def train(
        net, target_net,
        memory, episodes,
        copy_ops, train_ops
        ):
    eps = EPS_INIT
    step = 0
    rewards = []
    max_step = 200 # TODO : configure this

    writer = tf.summary.FileWriter(os.path.join(dirs['run_log_root'], 'train'), sess.graph)
    tf.summary.scalar('loss', net['loss'])
    summary = tf.summary.merge_all()

    c0 = np.zeros([1, N_H])
    h0 = np.zeros([1, N_H])

    def train_once(eps, step):
        s0 = proc(env.reset())

        # initialize episode
        net_reward = 0
        d = False
        c = c0.copy()
        h = h0.copy()
        entry = []

        for j in range(max_step):
            s0_1 = np.expand_dims(s0, 0)
            if (np.random.random() < eps) or (step < N_PRE):
                c, h = run(sess, net._tensors, ['x_in', 'c_in', 'h_in', 'n_b', 'n_t'],
                        [s0_1, c, h, 1, 1], ['c_out', 'h_out'])
                a = env.action_space.sample()
            else:
                a, c, h = run(sess, net._tensors, ['x_in', 'c_in', 'h_in', 'n_b', 'n_t'],
                        [s0_1, c, h, 1, 1], ['a_y', 'c_out', 'h_out'])
                a = a[0]

            s1,r,d,_ = env.step(a)
            s1 = proc(s1)

            entry.append(s0 + [a,r] + s1 + [d]) #[0:2, 2, 3, 4:6, 6]

            if step > N_PRE:
                if eps > EPS_F:
                    eps = max(EPS_MIN, eps * EPS_DECAY)
                if step % U_FREQ == 0:
                    sess.run(copy_ops) # update ...

                    input_batch = memory.sample(N_BATCH, N_TRACE)

                    x_in, a_in, r_in, _, d_in = np.split(input_batch, np.cumsum([N_X,1,1,N_X]), axis=-1)

                    x_in = np.reshape(x_in, [-1, N_X])
                    a_in = np.reshape(a_in, [-1])
                    r_in = np.reshape(r_in, [-1])
                    d_in = np.reshape(d_in, [-1])

                    c_in = np.zeros([N_BATCH, N_H])
                    h_in = np.zeros([N_BATCH, N_H])

                    a1, = run(sess, net._tensors, 
                            ['x_in', 'c_in', 'h_in', 'n_b', 'n_t'],
                            [x_in, c_in, h_in, N_BATCH, N_TRACE],
                            ['a_y']
                            ) # returns action-selection indices
                    q2, = run(sess, target_net._tensors, 
                            ['x_in', 'c_in', 'h_in', 'n_b', 'n_t'],
                            [x_in, c_in, h_in, N_BATCH, N_TRACE],
                            ['q_y']
                            ) # returns action assessment

                    qq = q2[range(N_BATCH*N_TRACE), a1] # "real" q values
                    q_t = r_in + GAMMA * qq * (1 - d_in) # discounted target q

                    # update ... 
                    s, _ = sess.run([summary, train_ops],
                            feed_dict = {
                                net['x_in'] : x_in,
                                net['c_in'] : c_in,
                                net['h_in'] : h_in,
                                net['n_b'] : N_BATCH,
                                net['n_t'] : N_TRACE,
                                net['q_t'] : q_t,
                                net['a_t'] : a_in
                                })
                    writer.add_summary(s, step)

            net_reward += r

            s0 = s1
            step += 1

            if d:
                break
        memory.add(np.asarray(entry))
        return net_reward, eps, step

    r_mean = 0
    i = 0
    sig = StopRequest()
    sig.start()

    while not sig._stop:
        net_reward, eps, step = train_once(eps, step)

        writer.add_summary(tf.Summary(value=[tf.Summary.Value(
            tag='net_reward',
            simple_value = net_reward
            )]), step)
        
        rewards.append(net_reward)

        if i % 100 == 0 and i > 0:
            r_mean = np.mean(rewards[-100:])
            r_max = np.max(rewards[-100:])
            print "[%d:%d] r(mean,max) (%.2f,%.2f) | Eps: %f" % (i, step, r_mean, r_max, eps)
        i += 1

    return rewards

def test(net, episodes):
    rewards = []

    # test
    for i in range(episodes):
        s = env.reset()
        d = False
        net_reward = 0
        while not d and net_reward < 200:
            env.render()
            a = sess.run(net.predict, feed_dict={net.inputs:[s]})
            a = a[0]
            s,r,d,_ = env.step(a)
            net_reward += r
        rewards.append(net_reward)
    return rewards

def main():
    # setup ... 
    drqn_a = DRQN([N_X], N_A, N_TRACE, scope='actor')
    drqn_c = DRQN([N_X], N_A, N_TRACE, scope='critic')
    memory = TraceMemory(size=N_MEM)

    # critic-update ...
    va = drqn_a.get_trainable_variables()
    vc = drqn_c.get_trainable_variables()
    copy_ops = [c.assign(a.value()*TAU + c.value() * (1.0-TAU)) for (a,c) in zip(va,vc)]
    copy_ops = tf.group(copy_ops)

    # train ...
    trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_ops = trainer.minimize(drqn_a._tensors['loss'])

    # initialize...
    sess.run(tf.global_variables_initializer())
    sess.run(copy_ops)
    saver = tf.train.Saver()

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'load':
        # load
        saver.restore(sess, '/tmp/model.ckpt')
        print '[loaded]'
    else:
        # train
        train_rewards = train(drqn_a, drqn_c, memory, N_EPOCH, copy_ops, train_ops)
        np.savetxt('train.csv', train_rewards, delimiter=',', fmt='%f')
        save_path = saver.save(sess, '/tmp/model.ckpt')
        print("Model saved in file: %s" % save_path) 

    test_rewards = test(drqn_a, N_TEST)
    np.savetxt('test.csv', test_rewards, delimiter=',', fmt='%f')

if __name__ == "__main__":
    main()
