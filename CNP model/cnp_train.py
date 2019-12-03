# [Conditional Neural Processes](https://arxiv.org/pdf/1807.01613.pdf) (CNPs)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
import cnp_preparedata
from sklearn.metrics import roc_curve, auc

#the datasets is encoded to npy files
rootpath = r'F:\yutian\Datasets\InterceptedDatasets\randomsample_encoding\npy'
CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))


# ## Conditional Neural Processes
#
# We can visualise a forward pass in a CNP as follows:
#
# <img src="https://bit.ly/2OFb6ZK" alt="drawing" width="400"/>
#
# As shown in the diagram, CNPs take in pairs **(x, y)<sub>i</sub>** of context
# points, pass them through an **encoder** to obtain
# individual representations **r<sub>i</sub>** which are combined using an **aggregator**. The resulting representation **r**
# is then combined with the locations of the targets **x<sub>T</sub>** and passed
# through a **decoder** that returns a mean estimate
# of the **y** value at that target location together with a measure of the
# uncertainty over said prediction. Implementing CNPs therefore involves coding up
# the three main building blocks:
#
# *   Encoder
# *   Aggregator
# *   Decoder
#
# A more detailed description of these three parts is presented in the following
# sections alongside the code.

# ## Encoder
class DeterministicEncoder(object):
    """The Encoder."""

    def __init__(self, output_sizes):
        self._output_sizes = output_sizes

    def __call__(self, context_x, context_y, num_context_points):

        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([context_x, context_y], axis=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = encoder_input.shape.as_list()
        hidden = tf.reshape(encoder_input, (batch_size * num_context_points, -1))
        hidden.set_shape((None, filter_size))

        # Pass through MLP
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            for i, size in enumerate(self._output_sizes[:-1]):
                hidden = tf.nn.relu(
                    tf.layers.dense(hidden, size, name="Encoder_layer_{}".format(i)))

            # Last layer without a ReLu
            hidden = tf.layers.dense(
                hidden, self._output_sizes[-1], name="Encoder_layer_{}".format(i + 1))

        # Bring back into original shape
        hidden = tf.reshape(hidden, (batch_size, num_context_points, size))

        # Aggregator: take the mean over all points
        representation = tf.reduce_mean(hidden, axis=1)

        return representation


# ## Decoder
class DeterministicDecoder(object):
    """The Decoder."""

    def __init__(self, output_sizes):
        self._output_sizes = output_sizes

    def __call__(self, representation, target_x, num_total_points):
        # Concatenate the representation and the target_x
        representation = tf.tile(
            tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
        input = tf.concat([representation, target_x], axis=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = input.shape.as_list()
        hidden = tf.reshape(input, (batch_size * num_total_points, -1))
        hidden.set_shape((None, filter_size))

        # Pass through MLP
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            for i, size in enumerate(self._output_sizes[:-1]):
                hidden = tf.nn.relu(
                    tf.layers.dense(hidden, size, name="Decoder_layer_{}".format(i)))

            # Last layer without a ReLu
            hidden = tf.layers.dense(
                hidden, self._output_sizes[-1], name="Decoder_layer_{}".format(i + 1))

        # Bring back into original shape
        hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))
        return hidden


# ## Model

class DeterministicModel(object):
    """The CNP model."""

    def __init__(self, encoder_output_sizes, decoder_output_sizes):

        self._encoder = DeterministicEncoder(encoder_output_sizes)
        self._decoder = DeterministicDecoder(decoder_output_sizes)

    def __call__(self, query, num_total_points, num_contexts, target_y=None):

        (context_x, context_y), target_x = query

        representation = self._encoder(context_x, context_y, num_contexts)
        output = self._decoder(representation, target_x, num_total_points)


        if target_y is not None:
            output = tf.reshape(output, [-1, 2])
            target_y = tf.reshape(target_y, [-1, 2])

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_y, logits=output))

            # accuracy
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            loss = None
            accuracy = None

        return loss, accuracy, output

TRAINING_ITERATIONS = 800
HIDDEN_SIZE = 128
MODEL_TYPE = 'NP'
ATTENTION_TYPE = 'uniform'
random_kernel_parameters = True
tf.reset_default_graph()
batchsize = 50

#Divide the training sets and tests set in a 1: 1 ratio
X_train, Y_train, X_test, Y_test = cnp_preparedata.get_train_test_data(rootpath)

# Train sets divided into context sets and target sets
context_X_train, context_Y_train, target_X_train, target_Y_train = cnp_preparedata.split_c_t(X_train, Y_train,
                                                                                             batch_size=batchsize)
context_X_train = tf.convert_to_tensor(context_X_train, dtype=tf.float32)
context_Y_train = tf.convert_to_tensor(context_Y_train, dtype=tf.float32)
target_X_train = tf.convert_to_tensor(target_X_train, dtype=tf.float32)
target_Y_train = tf.convert_to_tensor(target_Y_train, dtype=tf.float32)
train_query = ((context_X_train, context_Y_train), target_X_train)
data_train = CNPRegressionDescription(query=train_query, target_y=target_Y_train,
                                      num_total_points=target_X_train.shape[1],
                                      num_context_points=context_X_train.shape[1])
print(data_train)

# Test sets divided into context sets and target sets
context_X_test, context_Y_test, target_X_test, target_Y_test = cnp_preparedata.split_c_t(X_test, Y_test,
                                                                                         batch_size=batchsize)
context_X_test = tf.convert_to_tensor(context_X_test, dtype=tf.float32)
context_Y_test = tf.convert_to_tensor(context_Y_test, dtype=tf.float32)
target_X_test = tf.convert_to_tensor(target_X_test, dtype=tf.float32)
target_Y_test = tf.convert_to_tensor(target_Y_test, dtype=tf.float32)
test_query = ((context_X_test, context_Y_test), target_X_test)
data_test = CNPRegressionDescription(query=test_query, target_y=target_Y_test, num_total_points=target_X_test.shape[1],
                                     num_context_points=context_X_test.shape[1])
print(data_test)

encoder_output_sizes = [128, 128, 128, 128]
decoder_output_sizes = [128, 128, 2]

# Define the model
model = DeterministicModel(encoder_output_sizes, decoder_output_sizes)

# Define the loss
loss, acc, y_pred_train = model(data_train.query, data_train.num_total_points,
                                data_train.num_context_points, data_train.target_y)

# Get the predicted mean and variance at the target points for the testing set
_, _, y_pred_test = model(data_test.query, data_test.num_total_points,
                          data_test.num_context_points)

# Set up the optimizer and train step
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()

MODEL_DIRECTORY = "cnp_model/cnp_model.ckpt"
saver = tf.train.Saver()
max_acc = 0


with tf.Session() as sess:
    sess.run(init)

    for it in range(TRAINING_ITERATIONS):
        sess.run([train_step])

        train_acc, loss_value, y_score = sess.run([acc, loss, y_pred_train])
        print("Iteration: %4d, train_loss:%f,train_acc:%f" % (it, loss_value, train_acc))

        if train_acc > max_acc:
            max_acc = train_acc
            best_y_score = y_score
            save_path = saver.save(sess, MODEL_DIRECTORY)
            print("Model updated and saved in file: %s" % save_path)

    # test
    saver.restore(sess, MODEL_DIRECTORY)
    y_test = sess.run([y_pred_test])
    y_test = np.array(y_test)
    y_test = y_test.reshape([-1, 2])
    target_y = data_test.target_y.eval().reshape([-1, 2])
    correct_prediction = np.equal(np.argmax(y_test, 1), np.argmax(target_y, 1))
    accuracy = np.mean(correct_prediction)
    print("Test_acc:%f " % accuracy)

    #AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(target_y[:, i], y_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print("auc of player0:%f" % roc_auc[0])
    print("auc of player1:%f" % roc_auc[1])
    # micro average
    fpr["micro"], tpr["micro"], _ = roc_curve(target_y.ravel(), y_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print("auc of micro:%f" % roc_auc["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='CNPs roc of micro (auc = {0:0.2f})'
                   ''.format(roc_auc["micro"]), color='green', linestyle='-', linewidth=1.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves and AUC values')
    plt.legend(loc="lower right")
    plt.show()
