import numpy as np
import tensorflow as tf

def lerp(a,b,w):
    return w*a + (1.-w)*b

def get_copy_ops(src, dst, tau):
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src.scope)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dst.scope)
    ops = []
    for sv, dv in zip(src_vars, dst_vars):
        ops.append(dv.assign(lerp(sv,dv,tau)))
    return ops

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
