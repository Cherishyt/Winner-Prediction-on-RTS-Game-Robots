import tensorflow as tf
import prepare_data
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import lstm_model
import random

tf.reset_default_graph()
#the datasets is encoded to npy files
rootpath = r'E:\研\yutian\Datasets\randomsample_one_hot\npy'
MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

training_epochs = 40
batchsize = 50
# Hyper Parameters
learning_rate = 1e-4  # learning rate
n_steps = 8  # LSTM steps
n_inputs = 8*39 #if use feature-count encoding, replace it with 8*7.
n_hiddens = 64
n_layers = 2  # LSTM layers
n_classes = 2


# data
def train():
    # Prepare data
    T_train, W_train, T_val, W_val, T_test, W_test = prepare_data.getalldata(rootpath)
    train_size = T_train.shape[0]
    total_batch = int(train_size / batchsize)

    # tensor placeholder

    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='x_input')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')
    batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')

    # weights and biases
    Weights = tf.Variable(tf.truncated_normal([n_hiddens, n_classes], stddev=0.1), dtype=tf.float32, name='W')
    tf.summary.histogram('output_layer_weights', Weights)

    biases = tf.Variable(tf.random_normal([n_classes]), name='b')
    tf.summary.histogram('output_layer_biases', biases)

    y_pred=lstm_model.RNN_LSTM(x,Weights,biases,keep_prob,batch_size)
    #print(y_pred)

    # loss
    with tf.name_scope('LOSS'):
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y))
    tf.summary.scalar('loss', loss)

    # optimizer
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    max_acc = 0
    # training
    for epoch in range(training_epochs):
        i=0
        for batch_T,batch_W in prepare_data.minibatches(inputs=T_train,targets=W_train,batch_size=batchsize, shuffle=True):

            _, train_loss, train_accuracy,summary = sess.run([train_op, loss, accuracy, merged_summary_op],
                                                     feed_dict={x: batch_T, y: batch_W, keep_prob: 0.5,
                                                                batch_size: batchsize})
            summary_writer.add_summary(summary, epoch * total_batch + i)

            if i % 10 == 0:
                print("Epoch:", '%4d,' % (epoch),
                      "batch_index %4d/%4d,training loss %.5f, training accuracy %.5f" % (
                          i, total_batch, train_loss, train_accuracy))
                score, validation_loss, validation_accuracy = sess.run([y_pred, loss, accuracy],
                                                                       feed_dict={x: T_val, y: W_val, keep_prob: 0.5,
                                                                                  batch_size: T_val.shape[0]})
                print("Epoch:", '%4d,' % (epoch),
                      "batch_index %4d/%4d, validation loss %.5f,validation accuracy %.5f" % (
                          i, total_batch, validation_loss, validation_accuracy))
                if validation_accuracy > max_acc:
                    max_acc = validation_accuracy
                    best_y_score = score
                    save_path = saver.save(sess, MODEL_DIRECTORY)
                    print("Model updated and saved in file: %s" % save_path)
        index=[i for i in range(train_size)]
        random.shuffle(index)
        X_train=X_train[index]
        Y_train=Y_train[index]
    print("Max acc: %.5f" % max_acc)
    print("Optimization Finished!")

    # Restore variables from disk
    saver.restore(sess, MODEL_DIRECTORY)
    y_final, test_accuarcy = sess.run([y_pred, accuracy],
                                      feed_dict={x: T_test, y: W_test, keep_prob: 1.0, batch_size: T_test.shape[0]})
    print("test accuracy for the stored model: %f" % test_accuarcy)

    sess.close()
    #AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(W_test[:, i], y_final[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print("auc of player0:%f" % roc_auc[0])
    print("auc of player1:%f" % roc_auc[1])

    fpr["micro"], tpr["micro"], _ = roc_curve(W_test.ravel(), y_final.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print("auc of micro:%f" % roc_auc["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='LSTM roc of micro (auc = {0:0.2f})'
                   ''.format(roc_auc["micro"]), color='green', linestyle='-', linewidth=1.5)
    '''
    plt.plot(fpr[0], tpr[0],
            label='CNNs roc of player0(auc = {0:0.2f})'
                  ''.format(roc_auc[0]),color='navy', linestyle=':', linewidth=1.5)

    plt.plot(fpr[1], tpr[1],
            label='LSTM roc of player1(auc = {0:0.2f})'
                  ''.format(roc_auc[1]), color='yellow', linestyle='--', linewidth=1.5)
    '''
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
