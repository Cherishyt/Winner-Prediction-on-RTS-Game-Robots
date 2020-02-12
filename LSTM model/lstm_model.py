import tensorflow as tf

n_steps=8
n_inputs=8*39
n_hiddens=64
n_layers=2

# RNN structure
def RNN_LSTM(x, Weights, biases,keep_prob,batch_size):
    x = tf.reshape(x, [-1, n_steps, n_inputs])

    def lstm_cell():
        cell = tf.contrib.rnn.LSTMCell(n_hiddens, reuse=tf.get_variable_scope().reuse)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)], state_is_tuple=True)

    _init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=_init_state, dtype=tf.float32, time_major=False)
    return tf.matmul(outputs[:,-1,:], Weights) + biases
