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

## Network/Meta Parameters
N_H = 64 # number of hidden units
U_FREQ = 16 # update frequency
N_BATCH = 32 # size of training batch
N_TRACE = 8

## Q-Learning Parameters
GAMMA = .99 #Discount factor.
N_EPOCH = 10000 #Total number of episodes to train network for.
N_TEST = 200 #Total number of episodes to train network for.
TAU = 0.001 #Amount to update target network at each step.

# Annealing Parameters
EPS_0  = 1 #Starting chance of random action
EPS_F  = 0.05 #Final chance of random action
N_ANNEAL = 200000 #How many steps of training to reduce startE to endE.
eps_delta = float(EPS_F - EPS_0) / N_ANNEAL

pre_train_steps = 50000 #Number of steps us

## Initialize Tensorflow
tf.reset_default_graph()
sess= tf.Session()

def run(sess, tensors, ins_t, ins_v, outs):
    return sess.run([tensors[e] for e in outs],
            feed_dict = {tensors[t]:v for t,v in zip(ins_t, ins_v)})

def proc(x):
    # remove velocity information
    # (x,v,t,w) -> (x,t)
    return [x[0], x[2]]

def train(
        net, target_net,
        memory, episodes,
        copy_ops, train_ops
        ):
    eps = EPS_0
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
                if eps > EPS_F:
                    eps += eps_delta
                if step % U_FREQ == 0:
                    sess.run(copy_ops) # update ...

                    input_batch = memory.sample(N_BATCH, N_TRACE)
                    x_in, a_in, r_in, _, d_in = np.split(input_batch, [2,3,4,6])
                    c_in = np.zeros([N_BATCH, N_H])
                    h_in = np.zeros([N_BATCH, N_H])

                    q1, = run(sess, net._tensors, 
                            ['x_in', 'c_in', 'h_in', 'n_b', n_t],
                            [x_in, c_in, h_in, N_BATCH, N_TRACE],
                            ['a_y']
                            ) # returns action-selection indices
                    q2, = run(sess, target_net._tensors, 
                            ['x_in', 'c_in', 'h_in', 'n_b', n_t],
                            [x_in, c_in, h_in, N_BATCH, N_TRACE],
                            ['q_y']
                            ) # returns action assessment

                    qq = q2[range(N_BATCH*N_TRACE), q1] # "real" q values
                    q_t = r_in + GAMMA * qq * (1 - d_in) # discounted target q

                    # update ... 
                    sess.run(train_ops,
                            feed_dict = {
                                net['x_in'] : x_in,
                                net['n_b'] : N_BATCH,
                                net['n_t'] : N_TRACE,
                                net['q_t'] : q_t,
                                net['a_t'] : a_in
                                })
            net_reward += r
            s0 = s1
            step += 1

            if d:
                break
        memory.add(np.asarray(entry))
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

def main():

    # setup ... 
    drqn_a = DRQN([2], 2, 8, scope='actor')
    drqn_c = DRQN([2], 2, 8, scope='critic')

    va = drqn_a.get_trainable_variables()
    vc = drqn_c.get_trainable_variables()
    memory = TraceMemory(size=10000)

    TAU = 0.001
    copy_ops = [c.assign(a.value()*TAU + c.value() * (1.0-TAU)) for (a,c) in zip(va,vc)]
    copy_ops = tf.group(copy_ops)

    trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_ops = trainer.minimize(drqn_a._tensors['loss'])

    # get this started...
    sess.run(tf.initialize_all_variables())
    sess.run(copy_ops)
    saver = tf.train.Saver()

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'load':
        saver.restore(sess, '/tmp/model.ckpt')
        print '[loaded]'
    else:
        train_rewards = train(net, target_net, memory, N_EPOCH, copy_ops, train_ops)
        np.savetxt('train.csv', train_rewards, delimiter=',', fmt='%f')
        save_path = saver.save(sess, '/tmp/model.ckpt')
        print("Model saved in file: %s" % save_path) 

    test_rewards = test(net, N_TEST)
    np.savetxt('test.csv', test_rewards, delimiter=',', fmt='%f')

if __name__ == "__main__":
    main()
    
