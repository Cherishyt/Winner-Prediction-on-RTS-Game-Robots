# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import partial


# Create model of CNN with slim api
def MSCNN(inputs, is_training=True):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding="SAME"):
        with tf.variable_scope("Mixed_5b"):
            with tf.variable_scope("Branch_0"):
                batch_0 = slim.conv2d(inputs, num_outputs=64, kernel_size=[1, 1],
                                      scope="Conv2d_0a_1x1")
                #print(batch_0)
                batch_0 = slim.conv2d(batch_0, num_outputs=32, kernel_size=[1, 1],
                                      scope="Conv2d_0b_1x1")
                #print(batch_0)
                batch_0 = slim.conv2d(batch_0, num_outputs=1, kernel_size=[1, 1],
                                      scope="Conv2d_0c_1x1")
                #print(batch_0)
            with tf.variable_scope("Branch_1"):
                batch_1 = slim.conv2d(inputs, num_outputs=64, kernel_size=[1, 1],
                                      scope="Conv2d_0a_1x1")
                #print(batch_1)
                batch_1 = slim.conv2d(batch_1, num_outputs=96, kernel_size=[3, 3],
                                      scope="Conv2d_0b_3x3")
                #print(batch_1)
                batch_1 = slim.conv2d(batch_1, num_outputs=96, kernel_size=[3, 3],
                                      scope="Conv2d_0c_3x3")
                #print(batch_1)
                batch_1 = slim.conv2d(batch_1, num_outputs=32, kernel_size=[3, 3],
                                      scope="Conv2d_0d_3x3")
                #print(batch_1)
                batch_1 = slim.conv2d(batch_1, num_outputs=1, kernel_size=[3, 3],
                                      scope="Conv2d_0e_3x3")
                #print(batch_1)
            with tf.variable_scope("Branch_2"):
                batch_2 = slim.conv2d(inputs, num_outputs=48, kernel_size=[1, 1],
                                      scope="Conv2d_0a_1x1")
                #print(batch_2)
                batch_2 = slim.conv2d(batch_2, num_outputs=64, kernel_size=[5, 5],
                                      scope="Conv2d_0b_5x5")
                #print(batch_2)
                batch_2 = slim.conv2d(batch_2, num_outputs=32, kernel_size=[5, 5],
                                      scope="Conv2d_0c_5x5")
                #print(batch_2)
                batch_2 = slim.conv2d(batch_2, num_outputs=1, kernel_size=[5, 5],
                                      scope="Conv2d_0d_5x5")
                #print(batch_2)
            with tf.variable_scope("Branch_3"):
                batch_3 = slim.avg_pool2d(inputs, kernel_size=[3, 3], scope="AvgPool_0a_3x3")
                #print(batch_3)
                batch_3 = slim.conv2d(batch_3, num_outputs=30, kernel_size=[1, 1],
                                      scope="Conv2d_0b_1x1")
                #print(batch_3)
                batch_3 = slim.conv2d(batch_3, num_outputs=1, kernel_size=[1, 1],
                                      scope="Conv2d_0c_1x1")
                #print(batch_3)

            net = tf.concat([batch_0, batch_1, batch_2, batch_3], 3)
            #print(net)

            net= slim.flatten(net, scope='flatten1')
            net = slim.fully_connected(net, 256, scope='fc1')
            #print(net)  # 256
            net = slim.fully_connected(net, 64, scope='fc2')
            #print(net)  # 64
            outputs = slim.fully_connected(net, 2, scope='fco')
            #print(outputs)
    return outputs
