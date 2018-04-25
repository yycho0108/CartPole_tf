#!/usr/bin/python
"""
main_rnn.py

@Author : Yoonyoung Cho
@Date : 04/19/2018

Description : 
    Modification of main.py to run CartPole with recurrent neural networks (LSTMs)
    with DRQN-like architecture, following a [tutorial](https://github.com/awjuliani/DeepRL-Agents/blob/master/Deep-Recurrent-Q-Network.ipynb).
    Note that the CartPole model does not receive velocity-related data as input, meaning that it should learn to model the system motion.

Notes : Currently the expeirments **do** receive velocity-related input.
"""

from utils import *

import gym
import os
import sys
import numpy as np
import tensorflow as tf

from memory import TraceMemory
from drqn import DRQN
import argparse

## Network/Meta Parameters
HS = [32,64]
N_X = 4 # size of input
N_A = 2 # size of action
N_H = HS[-1]# number of hidden units
N_LOG = 16
N_BATCH = 32 # size of training batch
N_TRACE = 4
N_SKIP = 1

U_FREQ = 8 * N_TRACE # update frequency

## Learning Rate Parameters
LR_MAX = 1e-4
LR_MIN = 1e-5
LR_DECAY_STEPS = 2000000

## Q-Learning Parameters
GAMMA = .99 #Discount factor.
N_EPOCH = np.inf #20000 #Total number of episodes to train network for.
N_TEST = 200 #Total number of episodes to train network for.
TAU = 1e-3#1e-3 #(1.0/100) * U_FREQ #Amount to update target network at each step.

# Exploration Parameters
EPS_INIT  = 1.0 #Starting chance of random action
EPS_MIN  = 0.05 #Final chance of random action
EPS_ANNEAL = 2000000 #How many steps of training to reduce startE to endE.
EPS_DECAY = EPS_MIN ** (1.0/EPS_ANNEAL)
#EPS_DECAY = 0.9999

N_PRE = 50000 #Number of steps, pre-train
N_MEM = 10000 # ~5000 episodes

GAME_STEPS = 999

PARAMS = {
        'N_X' : N_X,
        'N_A' : N_A,
        'N_H' : N_H,
        'U_FREQ' : U_FREQ,
        'N_BATCH' : N_BATCH,
        'N_TRACE' : N_TRACE,
        'LR_MAX' : LR_MAX,
        'LR_MIN' : LR_MIN,
        'LR_DECAY_STEPS' : LR_DECAY_STEPS,
        'GAMMA' : GAMMA,
        'N_EPOCH' : N_EPOCH,
        'N_TEST' : N_TEST,
        'TAU' : TAU,
        'EPS_INIT' : EPS_INIT,
        'EPS_MIN' : EPS_MIN,
        'EPS_ANNEAL' : EPS_ANNEAL,
        'EPS_DECAY' : EPS_DECAY,
        'N_PRE' : N_PRE,
        'N_MEM' : N_MEM,
        'HS' : HS
        }

def proc(x):
    # remove velocity information
    # (x,v,t,w) -> (x,t)
    #return [x[0], x[2]]
    # decompose t -> sin(t), cos(t)
    x[1] = x[3] = 0.0
    return list(x)

def get_eps(x):
    if x < 1e6: # 1->0.1 in 1M steps
        # from (0 - 1e6) 1 -> 0.1
        return 1.0 * 0.1**(x*1.0/1e6)
    elif x < 1e7: # 0.1 -> 0.01 in 9M steps
        # from (1e6 - 1e7) 0.1 -> 0.05
        return 0.1 * 0.5**((x-1e6) * 1.0 / 9e6)
    elif x < 2e7:
        #from (1e7-2e7) 0.05 -> 0.01
        return 0.05 * 0.2**((x-1e7)*1.0/1e7)
    else:
        return 0.01

class DRQNMain(object):

    def __init__(self, env, name):
        self._env = env
        #PARAMS['NAME'] = name
        self._dirs = directory_setup('drqn', run_id=name, **PARAMS)
        self._build()
        
        gpu_options = tf.GPUOptions(
                allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False,
                gpu_options=gpu_options)
        self._sess = tf.Session(config=config)
        self.reset()

    def _build(self):
        tf.reset_default_graph()
        # setup ... 
        drqn_a = DRQN([N_X], N_A, N_TRACE, hs=HS, scope='actor')
        drqn_c = DRQN([N_X], N_A, N_TRACE, hs=HS, scope='critic')
        memory = TraceMemory(size=N_MEM)

        # critic-update ...
        with tf.name_scope('copy'):
            va = drqn_a.get_trainable_variables()
            vc = drqn_c.get_trainable_variables()
            copy_ops = [c.assign(a.value()*TAU + c.value()*(1.0-TAU)) for (a,c) in zip(va,vc)]
            #copy_ops = [c.assign(a.value()) for (a,c) in zip(va,vc)]
            copy_ops = tf.group(copy_ops)

        # train ...
        with tf.name_scope('train'):
            tf_step = tf.Variable(0, trainable=False)
            lr = tf.maximum(tf.train.exponential_decay(LR_MAX, tf_step, LR_DECAY_STEPS, LR_MIN/LR_MAX), LR_MIN)
            trainer = tf.train.AdamOptimizer(learning_rate=lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                g = trainer.compute_gradients(drqn_a['loss'], var_list=va)
                train_ops = trainer.apply_gradients(g)
                gn = tf.global_norm(g)
                #train_ops = trainer.minimize(
                #        drqn_a['loss'],
                #        var_list = va
                #        )
        graph = tf.get_default_graph()    
        writer = tf.summary.FileWriter(os.path.join(self._dirs['run_log_root'], 'train'), graph)

        with tf.name_scope('logs'):
            tf.summary.scalar('learning_rate', lr)
            tf.summary.scalar('q', tf.reduce_mean(drqn_a['q']))
            tf.summary.scalar('q_t', tf.reduce_mean(drqn_a['q_t']))
            tf.summary.scalar('loss', drqn_a['loss'])
            tf.summary.scalar('q_s', drqn_a['q_s'])
            tf.summary.scalar('g', gn)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        
        self._drqn_a = drqn_a
        self._drqn_c = drqn_c
        self._memory = memory
        self._copy_ops = copy_ops
        self._train_ops = train_ops
        self._writer = writer
        self._summary = summary
        self._tf_step = tf_step
        self._saver = saver

    def act(self, s, c, h):
        env  = self._env
        step = self._step
        eps  = self._eps
        net = self._drqn_a

        # add batch size
        s = np.expand_dims(s, 0)

        feed_dict = {
                net['x_in'] : s,
                net['c_in'] : c,
                net['h_in'] : h,
                net['n_b'] : 1,
                net['n_t'] : 1
                }

        if (np.random.random() < eps) or (step < N_PRE):
            c, h = self.run(
                    [net['c_out'], net['h_out']],
                    feed_dict = feed_dict)
            a = env.action_space.sample()
        else:
            a, c, h = self.run(
                    [net['a_y'], net['c_out'], net['h_out']],
                    feed_dict = feed_dict)
            a = a[0]

        return a, c, h

    def step(self, s0, c, h):
        """ steps once, returns step entry """
        env = self._env

        a, c, h = self.act(s0, c, h)

        for i in range(N_SKIP):
            s1, r, d, _ = env.step(a)
            if d:
                break

        #r = float(r)
        #s = xvtw
        r = np.cos(5*s1[2]) # theta reward
        #r = r * (1.0 - s1[0] / 2.4) # x reward
        #r = r * (1.0-d)
        #if d:
        #    r = -100.0
        # TODO : engineered reward! danger.
        s1 = proc(s1)
        self._step += 1
        return s1, a, r, d, c, h

    def update(self):
        net_a, net_c = self._drqn_a, self._drqn_c

        # unroll ...
        memory = self._memory
        writer = self._writer
        summary = self._summary
        tf_step = self._tf_step
        step = self._step
        train_ops = self._train_ops
        copy_ops = self._copy_ops

        input_batch = memory.sample(N_BATCH, N_TRACE)
        x0_in, a_in, r_in, x1_in, d_in = np.split(input_batch, np.cumsum([N_X,1,1,N_X]), axis=-1)

        x0_in = np.reshape(x0_in, [-1, N_X])
        a_in = np.reshape(a_in, [-1])
        r_in = np.reshape(r_in, [-1])
        x1_in = np.reshape(x1_in, [-1, N_X])
        d_in = np.reshape(d_in, [-1])
        c_in = np.zeros([N_BATCH, N_H])
        h_in = np.zeros([N_BATCH, N_H])

        a, q = self.run([net_a['a_y'], net_c['q_y']], feed_dict={
            net_a['x_in'] : x1_in,
            net_a['c_in'] : c_in,
            net_a['h_in'] : h_in,
            net_a['n_b'] : N_BATCH,
            net_a['n_t'] : N_TRACE,
            net_c['x_in'] : x1_in,
            net_c['c_in'] : c_in,
            net_c['h_in'] : h_in,
            net_c['n_b'] : N_BATCH,
            net_c['n_t'] : N_TRACE
            })

        q = q[range(N_BATCH*N_TRACE), a]
        q_t = r_in + GAMMA * q * (1.0 - d_in)

        s, _ = self.run([summary, train_ops],
                feed_dict = {
                    net_a['x_in'] : x0_in,
                    net_a['c_in'] : c_in,
                    net_a['h_in'] : h_in,
                    net_a['n_b'] : N_BATCH,
                    net_a['n_t'] : N_TRACE,
                    net_a['q_t'] : q_t,
                    net_a['a_t'] : a_in,
                    tf_step : (step - N_PRE)
                    })
        writer.add_summary(s, step)
        self.run(copy_ops)

    def train_1(self, c0, h0, max_step):
        """ train for one episode """
        env = self._env
        s0 = proc(env.reset())
        entry = []
        net_reward = 0.
        d = False
        c = c0.copy()
        h = h0.copy()

        for i in range(0, max_step, N_SKIP):
            s1, a, r, d, c, h = self.step(s0, c, h)
            entry.append(s0 + [a,r] + s1 + [d])
            if self._step > N_PRE:
                #self._eps = max(EPS_MIN, self._eps * EPS_DECAY)
                self._eps = get_eps(self._step - N_PRE)
                if (self._step % U_FREQ) == 0:
                    self.update()
            net_reward += r
            s0 = s1
            if d:
                break

        return net_reward, entry

    def reset(self):
        self._eps = EPS_INIT
        self._step = 0
        self.run(tf.global_variables_initializer())
        self.run(self._copy_ops)

    def add_summary(self, value, tag):
        self._writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag=tag,simple_value=value)
            ]), self._step)

    def train(self, n):
        """ train for n episodes """
        rewards = []
        c0 = np.zeros([1, N_H])
        h0 = np.zeros([1, N_H])

        i = 0 # episode index
        sig = StopRequest()
        sig.start()

        while not sig._stop:
            # TODO : hardcoded max_step
            net_reward, entry = self.train_1(c0, h0, GAME_STEPS-2)
            #if self._step > N_PRE:
            #    self.update()
            if len(entry) >= N_TRACE:
                self._memory.add(np.asarray(entry))

            rewards.append(net_reward)

            if i % 100 == 0 and i > 0:
                rw = rewards[-100:]

                r_mean = np.mean(rw)
                r_min = np.min(rw)
                r_max = np.max(rw)

                self.add_summary('r_mean', r_mean)
                self.add_summary('r_min', r_min)
                self.add_summary('r_max', r_max)
                self.add_summary('eps', self._eps)

                print "[%d:%d] r(mean,max) (%.2f,%.2f) | Eps: %f" % (i, self._step, r_mean, r_max, self._eps)
            i += 1

        sig.stop() # i.e. stop handling signals
        return rewards

    def test(self, n):
        """ test for n episodes """
        env = self._env
        net = self._drqn_a

        rewards = []
        c0 = np.zeros([1, N_H])
        h0 = np.zeros([1, N_H])
        sig = StopRequest()
        sig.start()

        for i in range(n):
            if sig._stop:
                break

            s = env.reset()
            d = False
            net_reward = 0
            c = c0.copy()
            h = h0.copy()

            while not d and net_reward < 999:
                env.render()
                x = np.expand_dims(proc(s), 0)
                a, c, h = self.run([net['a_y'], net['c_out'], net['h_out']],
                        feed_dict={
                            net['x_in'] : x,
                            net['c_in'] : c,
                            net['h_in'] : h,
                            net['n_b'] : 1,
                            net['n_t'] : 1
                            })
                s,r,d,_ = env.step(a[0])
                net_reward += r

            print i, net_reward

            rewards.append(net_reward)

        sig.stop()
        return rewards


    def save(self):
        path = self._dirs['output_ckpt']
        save_path = self._saver.save(self._sess, path)
        print("Model saved in file: %s" % save_path)

    def load(self, path='/tmp/model.ckpt'):
        ## TODO : actually not ignore path
        self._saver.restore(self._sess, path)
        print("Model loaded from file: %s" % path)

    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(opts):
    gym.envs.register(
            id='CartPole-v0999',
            entry_point='gym.envs.classic_control:CartPoleEnv',
            tags={'wrapper_config.TimeLimit.max_episode_steps': GAME_STEPS},
            reward_threshold=float(GAME_STEPS)-1.0,
            )
    env = gym.make('CartPole-v0999')
    env.reset()

    #while True:
    #    env.render()
    #    s1, r, d, k = env.step(0)
    #    print d
    #    if d:
    #        env.reset()

    app = DRQNMain(env, opts.name)

    if opts.load:
        # load
        app.load(opts.load)

    if opts.train:
        # train
        train_rewards = app.train(N_EPOCH)
        np.savetxt('train.csv', train_rewards, delimiter=',', fmt='%f')
        app.save()

    if opts.test:
        test_rewards = app.test(N_TEST)
        np.savetxt('test.csv', test_rewards, delimiter=',', fmt='%f')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DRQN on Cartpole.')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--test', type=str2bool, default=False)

    args = parser.parse_args()
    if args.train and not args.name:
        parser.error("--train requires --name!")
    main(args)
