import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import slim

class DRQN(object):
    def __init__(self,
            state_shape,
            n_action,
            n_steps,
            scope='drqn', data_format='NCHW',
            is_training=True, reuse=None
            ):
        self._state_shape = state_shape
        self._n_action = n_action
        self._n_steps = n_steps
        self._hs = [8, 32, 64] # currently not configurable
        self._scope = scope
        self._data_format = data_format
        self._is_training = is_training
        self._reuse = reuse
        self._build()

    def _arg_scope(self):
        batch_norm_params = {
                'is_training' : self._is_training,
                'decay' : 0.995,
                'fused' : True,
                'scale' : True,
                'reuse' : self._reuse,
                'data_format' : self._data_format,
                'scope' : 'batch_norm',
                }
        with slim.arg_scope([slim.fully_connected],
                activation_fn = tf.nn.elu,
                #normalizer_fn = slim.batch_norm,
                #normalizer_params = batch_norm_params,
                # don't use batch norm, for complicated reasons
                ) as sc:
            return sc

    def _build_fcn(self, x):
        # [batch_size, 2] -> [batch_size, 64]
        with tf.name_scope('fcn', [x]):
            with slim.arg_scope(self._arg_scope()):
                # x = (cart position, cart angle) = 2
                return slim.stack(x, slim.fully_connected, self._hs, scope='fc')
            # 2  (2x8) 8 (8x32) 32 (32x64) 64 (64x2) 2

    def _build_cnn(self, x):
        with tf.name_scope('cnn', [x]):
            with slim.arg_scope(self._arg_scope()):
                pass

    def _build_rnn(self, x, b, s):
        # TODO : treat conv vs. fc differently somehow?
        with tf.name_scope('rnn', [x]):
            n_h = self._hs[-1]
            x = slim.flatten(x) # in case input is conv.
            x = tf.reshape(x, [b, s, n_h])
            with slim.arg_scope(self._arg_scope()):
                cell = rnn.BasicLSTMCell(n_h, state_is_tuple=True)
                s0 = cell.zero_state(b, tf.float32)
                #c_in = tf.placeholder_with_default(
                #        s0.c,
                #        s0.c.shape,
                #        name='c_in'
                #        )
                #h_in = tf.placeholder_with_default(
                #        s0.h,
                #        s0.h.shape,
                #        name='h_in'
                #        )
                c_in = tf.placeholder(
                        shape = s0.c.shape,
                        dtype = tf.float32,
                        name = 'c_in')
                h_in = tf.placeholder(
                        shape = s0.h.shape,
                        dtype = tf.float32,
                        name = 'h_in')
                s_in = rnn.LSTMStateTuple(c_in, h_in)
                y, s_out = tf.nn.dynamic_rnn(
                        inputs=x,
                        cell=cell,
                        dtype=tf.float32,
                        initial_state=s_in,
                        scope='rnn')
                # y = (batch_size, n_steps, n_actions)
                #y = y[:,-1,:]
                y = tf.reshape(y, [-1, n_h])
                # TODO : why is this ok?
                # apparently q is computed for each time step as well. hmm

                c_out = s_out.c
                h_out = s_out.h
                #cell = rnn.Conv2DLSTMCell(
                #        input_shape = (?)
                #        output_channels = self._n_action,
                #        kernel_shape = (3,3),
                #        use_bias=True,
                #        skip_connection=False,
                #        initializer=slim.initializers.xavier_initializer()
                #        )
        return c_in, h_in, y, c_out, h_out

    def _build_qn(self, x, n_b, n_t):
        """ Build Q-Network """
        with tf.name_scope('qn', [x]):
            with slim.arg_scope(self._arg_scope()):
                sa, sv = tf.split(x, 2, axis=1) # split into action-value streams
                adv = slim.fully_connected(sa, self._n_action, scope='adv')
                val = slim.fully_connected(sv, 1, scope='val')
                q_y = val + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))
                a_y = tf.argmax(q_y, axis=1)

                # setup targets
                # TODO : add eval flag to enable creating loss/evaluation targets
                q_t = tf.placeholder(shape=[None], dtype=tf.float32, name='q_t')
                a_t = tf.placeholder(shape=[None], dtype=tf.int32, name='a_t')
                a_t_o = tf.one_hot(a_t, self._n_action, dtype=tf.float32)

                q = tf.reduce_sum(q_y * a_t_o, axis=1)
                q_err = tf.square(q_t - q)

                m_a = tf.zeros([n_b, n_t//2], dtype=tf.float32)
                m_b = tf.ones([n_b, n_t//2], dtype=tf.float32)
                mask = tf.concat([m_a, m_b], 1)
                mask = tf.reshape(mask, [-1])
                loss = tf.reduce_mean(q_err * mask)
                #loss = tf.reduce_mean(q_err)

        return q_t, a_t, q_y, a_y, loss


    def _build(self):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            x_in = tf.placeholder(tf.float32, shape=[None] + list(self._state_shape), name='x_in')
            batch_size = tf.placeholder(tf.int32, shape=[], name='n_b')
            step_size = tf.placeholder(tf.int32, shape=[], name='n_t')
            #cnn = self._build_cnn(self._inputs)
            fcn = self._build_fcn(x_in)
            c_in, h_in, y, c_out, h_out = self._build_rnn(fcn, batch_size, step_size)
            q_t, a_t, q_y, a_y, loss = self._build_qn(y, batch_size, step_size)

        # save ; inputs
        self._inputs = {
                'n_b' : batch_size,
                'n_t' : step_size,
                'x_in' : x_in, # actual env input
                'c_in' : c_in, # states bookkeeping
                'h_in' : h_in, #
                'q_t'  : q_t,  # target q
                'a_t'  : a_t   # target action
                }
        self._outputs = {
                'y' : y, # rnn output
                'c_out' : c_out, # states bookkeeping
                'h_out' : h_out, # 
                'q_y' : q_y,     # network q
                'a_y' : a_y,      # network action
                'loss' : loss
                }

        # create tensors lookup dictionary
        self._tensors = self._inputs.copy()
        self._tensors.update(self._outputs)

    def __getitem__(self, name):
        return self._tensors[name]

    def predict(self):
        return NotImplementedError("DRQN.predict() does not exist yet.")

    def train(self):
        return NotImplementedError("DRQN.train() does not exist yet.")

    def get_trainable_variables(self):
        return slim.get_trainable_variables(self._scope)
        

def main():
    # test ...
    drqn_a = DRQN([2], 2, 8, scope='actor')
    drqn_c = DRQN([2], 2, 8, scope='critic')

    va = drqn_a.get_trainable_variables()
    vc = drqn_c.get_trainable_variables()

    tau = 0.001
    copy_ops = [c.assign(a.value()*tau + c.value() * (1.0-tau)) for (a,c) in zip(va,vc)]
    copy_ops = tf.group(copy_ops)

    print drqn_a['x_in']


if __name__ == "__main__":
    main()
