# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import roc_curve,auc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import prepare_data
import cnn_model
import numpy as np
import random

MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"
#the datasets is encoded to npy files
rootpath=r'E:\研\yutian\Datasets\randomsample_one_hot\npy'

# Params for Train
training_epochs = 40
batch_size = 50
display_step = 10
validation_step = 10

def train():
    # Prepare data
    T_train, W_train, T_val, W_val,T_test,W_test = prepare_data.getalldata(rootpath)
    train_size=T_train.shape[0]

    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 8,8,39])
    y_ = tf.placeholder(tf.float32, [None, 2])

    # Predict
    y = cnn_model.CNN(x,is_training)
    # Get loss of model
    with tf.name_scope("LOSS"):
        loss = slim.losses.softmax_cross_entropy(y,y_)
    print(loss)
    # Create a summary to monitor loss tensor
    tf.summary.scalar('loss', loss)

    # Define optimizer
    with tf.name_scope("ADAM"):
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(
            1e-4,  # Base learning rate.
            batch * batch_size,  # Current index into the dataset.
            train_size,  # Decay step.
            0.95,  # Decay rate.
        staircase=True)
        # Use simple momentum for the optimization.
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)

    # Create a summary to monitor learning_rate tensor
    tf.summary.scalar('learning_rate', learning_rate)

    # Get accuracy of model
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Training cycle
    total_batch = int(train_size / batch_size)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # Save the maximum accuracy value for validation data
    max_acc = 0.

    # Loop for epoch
    for epoch in range(training_epochs):

        # Loop over all batches
        for batch_T, batch_W in prepare_data.minibatches(T_train, W_train, batch_size, shuffle=True):
            i=0
            _, train_loss, train_accuracy, summary = sess.run([train_step, loss, accuracy, merged_summary_op] , feed_dict={x: batch_T, y_: batch_W, is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print("Epoch:", '%4d,' % (epoch),
                "batch_index %4d/%4d,training loss %.5f, training accuracy %.5f" % (i, total_batch,train_loss, train_accuracy))

            # Get accuracy for validation data
            if i % validation_step == 0:
                # Calculate accuracy
                score, validation_loss, validation_accuracy = sess.run([y, loss, accuracy],
                feed_dict={x: T_val, y_: W_val, is_training: False})

                print("Epoch:", '%4d,' % (epoch ),
                "batch_index %4d/%4d, validation loss %.5f,validation accuracy %.5f" % (i, total_batch,validation_loss, validation_accuracy))
            i+=1
            # Save the current model if the maximum accuracy is updated
            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                #best_y_score=score
                save_path = saver.save(sess, MODEL_DIRECTORY)
                print("Model updated and saved in file: %s" % save_path)
        #打乱
        index=[i for i in range(train_size)]
        random.shuffle(index)
        X_train=X_train[index]
        Y_train=Y_train[index]
    print("Max acc: %.5f" % max_acc)
    print("Optimization Finished!")


    # Restore variables from disk
    saver.restore(sess, MODEL_DIRECTORY)
    y_final = sess.run(y, feed_dict={x: T_test, y_: W_test, is_training: False})
    correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(W_test, 1))

    print("test accuracy for the stored model: %f" % np.mean(correct_prediction))
    sess.close()

    #AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i],tpr[i],_ = roc_curve(W_test[:,i], y_final[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])

    print("auc of player0:%f"%roc_auc[0])
    print("auc of player1:%f"%roc_auc[1])
    # micro average
    fpr["micro"], tpr["micro"], _ = roc_curve(W_test.ravel(), y_final.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print("auc of micro:%f"%roc_auc["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='CNNs roc of micro (auc = {0:0.2f})'
                  ''.format(roc_auc["micro"]), color='green', linestyle='-', linewidth=1.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves and AUC values')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    train()
