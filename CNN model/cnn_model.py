import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import partial
# Create model of CNN with slim api
def CNN(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):

        net = slim.conv2d(inputs, 32, [5, 5], scope='conv1')
        #print(net)#8*8*32
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        #print(net)#4*4*32
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        #print(net)#4*4*64
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        #print(net)#2*2*64
        net = slim.flatten(net, scope='flatten3')
        #print(net)#256
        net = slim.fully_connected(net, 1024, scope='fc3')
        #print(net)#1024
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
        #print(net)
        outputs = slim.fully_connected(net, 2, activation_fn=None, normalizer_fn=None, scope='fco')
        #print(outputs)

    return outputs
