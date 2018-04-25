import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import slim

def huber_loss(y, t, delta=1.0):
    with tf.name_scope('huber_loss', [y, t]):
        err = tf.abs(y - t)
        q = tf.minimum(err, delta)
        return 0.5 * tf.square(q) + delta * (err - q)


class DRQN(object):
    def __init__(self,
            state_shape,
            n_action,
            n_steps,
            hs=[32,64],
            scope='drqn', data_format='NCHW',
            is_training=True, reuse=None
            ):
        self._state_shape = state_shape
        self._n_action = n_action
        self._n_steps = n_steps
        self._hs = hs # currently not configurable
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
                #weights_regularizer=slim.l2_regularizer(1e-4)
                #normalizer_fn = slim.batch_norm,
                #normalizer_params = batch_norm_params,
                # don't use batch norm, for complicated reasons
                ) as sc:
            return sc

    def _build_fcn(self, x):
        # [batch_size, 2] -> [batch_size, 64]
        with tf.name_scope('fcn', [x]):
            with slim.arg_scope(self._arg_scope()):
                return slim.stack(x, slim.fully_connected, self._hs, scope='fc')

    def _build_cnn(self, x):
        with tf.name_scope('cnn', [x]):
            with slim.arg_scope(self._arg_scope()):
                return NotImplementedError("CNN Not Supported Yet!")

    def _build_rnn(self, x, b, s):
        # TODO : treat conv vs. fc differently somehow?
        with tf.name_scope('rnn', [x]):
            n_h = self._hs[-1]
            x = slim.flatten(x) # in case input is conv.
            x = tf.reshape(x, [b, s, n_h])
            with slim.arg_scope(self._arg_scope()):
                cell = rnn.BasicLSTMCell(n_h, state_is_tuple=True)
                s0 = cell.zero_state(b, tf.float32)
                #c_in = s0.c
                #h_in = s0.h
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
                xf = slim.fully_connected(x, 128,
                        scope='xf', activation_fn=tf.nn.elu)
                sa, sv = tf.split(xf, 2, axis=1) # split into action-value streams
                adv = slim.fully_connected(sa, 64,
                        scope='adv_0', activation_fn=tf.nn.elu)
                adv = slim.fully_connected(adv, self._n_action,
                        scope='adv_1', activation_fn=None)

                val = slim.fully_connected(sv, 64,
                        scope='val_0', activation_fn=tf.nn.elu)
                val = slim.fully_connected(val, 1,
                        scope='val_1', activation_fn=None)

                q_y = val + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))
                a_y = tf.argmax(q_y, axis=1)
        return q_y, a_y

    def _build_loss(self, q_y, a_y, n_b, n_t):
        with tf.name_scope('err', [q_y, a_y]):
            # setup targets
            # TODO : add eval flag to enable creating loss/evaluation targets
            q_t = tf.placeholder(shape=[None], dtype=tf.float32, name='q_t')
            a_t = tf.placeholder(shape=[None], dtype=tf.int32, name='a_t')
            a_t_o = tf.one_hot(a_t, self._n_action, dtype=tf.float32)

            q = tf.reduce_sum(q_y * a_t_o, axis=1)

            decay = 0.99

            q_m = tf.Variable(initial_value=1.0, trainable=False)
            q_s = tf.Variable(initial_value=1.0, trainable=False)

            #q_t_m, q_t_v = tf.nn.moments(q_t, axes=[0])
            #q_t_s = tf.sqrt(q_t_v)

            #u_m = q_m.assign(q_m*decay + q_t_m*(1.0-decay)) # offset
            #u_s = q_s.assign(q_s*decay + q_t_s*(1.0-decay)) # scale
            #with tf.control_dependencies([u_m, u_s]):
            #    q_n = (q - q_m) / (2.0 * q_s)
            #    q_t_n = (q_t - q_m) / (2.0 * q_s)
            #    q_err = huber_loss(q_n, q_t_n)

            # OPT1 . relative error
            #q_s = tf.Variable(initial_value=1.0, trainable=False)
            #q_t_s = tf.reduce_mean(q_t)

            #with tf.control_dependencies([q_s.assign(q_s*q_s_decay + q_t_s*(1.0-q_s_decay))]):
            #    q_n = q / q_s
            #    q_t_n = q_t / q_s
            #    q_err = huber_loss(q_n, q_t_n)

            # OPT2.0 absolute error huber
            q_err = huber_loss(q, q_t)# / tf.square(tf.reduce_mean(q_t))

            # OPT2.1 absolute error square
            #q_err = tf.square(q_t-q)

            # only the latter steps will be counted for loss ...
            n_mask = tf.maximum(n_t//2, 1)
            m_a = tf.zeros([n_b, n_t - n_mask], dtype=tf.float32)
            m_b = tf.ones([n_b, n_mask], dtype=tf.float32)
            mask = tf.concat([m_a, m_b], 1)
            mask = tf.reshape(mask, [-1])
            loss = tf.reduce_mean(q_err * mask)

        return q, q_t, a_t, loss, q_s

    #def _popart(self, y, t):
    #    with tf.name_scope('popart', [y,t]):
    #        t_n = tf.matmul(tf.inv(sigma), y - mu)
    #        # y = [n_b]
    #        #W = tf.eye(...)

    def _build(self):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            x_in = tf.placeholder(tf.float32, shape=[None] + list(self._state_shape), name='x_in')
            batch_size = tf.placeholder(tf.int32, shape=[], name='n_b')
            step_size = tf.placeholder(tf.int32, shape=[], name='n_t')
            #cnn = self._build_cnn(self._inputs)
            fcn = self._build_fcn(x_in)
            c_in, h_in, y, c_out, h_out = self._build_rnn(fcn, batch_size, step_size)
            q_y, a_y = self._build_qn(y, batch_size, step_size)
            q, q_t, a_t, loss, q_s = self._build_loss(q_y, a_y, batch_size, step_size)


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
                'q' : q,
                'q_y' : q_y,     # network q
                'a_y' : a_y,      # network action
                'loss' : loss,
                'q_s' : q_s
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


if __name__ == "__main__":
    main()
