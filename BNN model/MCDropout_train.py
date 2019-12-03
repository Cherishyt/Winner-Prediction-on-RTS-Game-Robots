import time
from src.MC_dropout.model import *
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import os

import matplotlib.pyplot as plt

rootpath=r'F:\yutian\Datasets\InterceptedDatasets\randomsample_encoding\npy'
models_dir = "MCdrop_model"
results_dir = "MCdrop_results"
mkdir(models_dir)
mkdir(results_dir)
# ------------------------------------------------------------------------------------------------------
# train config
batch_size = 50
nb_epochs = 100
lr = 1e-4

# ------------------------------------------------------------------------------------------------------
# load data
X,Y=[],[]
for file in os.listdir(rootpath):
    filename=os.path.splitext(file)[0]
    #get labels
    winner=int(filename[-1:])
    if winner==0:
        Y.append([1,0])
    elif winner==1:
        Y.append([0,1])
    else:
        raise ValueError("Invalid winner value!")
    x=np.load(os.path.join(rootpath,file))
    X.append(x)

X = np.array(X)  #(,8,8,38)
Y = np.array(Y)  #(,2)
print(X.shape)
print(Y.shape)

# 50% train sets，25% val sets，25% test sets
X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=1 / 2)
X_test, X_val, Y_test, Y_val = train_test_split(X_val_test, Y_val_test, test_size=1 / 2)
print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)

NTrainPoints = X_train.shape[0]
print(NTrainPoints)
use_cuda = torch.cuda.is_available()

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c', '\nNetwork:')

net = MC_drop_net(lr=lr, channels_in=38, side_in=8, cuda=use_cuda, classes=2, batch_size=batch_size,
                  weight_decay=1, n_hid=1200)

## ---------------------------------------------------------------------------------------------------------------------
# train
epoch = 0
cprint('c', '\nTrain:')

print('  init cost variables:')
kl_cost_train = np.zeros(nb_epochs)
pred_cost_train = np.zeros(nb_epochs)
err_train = np.zeros(nb_epochs)
acc_train = np.zeros(nb_epochs)

cost_dev = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)
acc_dev = np.zeros(nb_epochs)
best_err = np.inf

nb_its_dev = 1

tic0 = time.time()
for i in range(epoch, nb_epochs):

    net.set_mode_train(True)
    tic = time.time()
    nb_samples = 0

    for x, y in minibatches(X_train, Y_train, batch_size, shuffle=True):
        cost_pred, err, acc = net.fit(x, y)

        err_train[i] += err
        pred_cost_train[i] += cost_pred
        acc_train[i] += acc
        nb_samples += len(x)

    pred_cost_train[i] /= nb_samples
    err_train[i] /= nb_samples
    acc_train[i] /= nb_samples

    toc = time.time()
    net.epoch = i
    # ---- print
    print("it %d/%d, Jtr_pred = %f,train_acc = %f " % (
    i, nb_epochs, pred_cost_train[i], acc_train[i]))
    cprint('r', '   time: %f seconds\n' % (toc - tic))

    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        nb_samples = 0
        cost, err, probs, acc, out = net.eval(X_val, Y_val)

        cost_dev[i] += cost
        err_dev[i] += err
        acc_dev[i] += acc
        nb_samples = len(X_val)

        cost_dev[i] /= nb_samples
        err_dev[i] /= nb_samples
        acc_dev[i] /= nb_samples

        cprint('g', '    Jdev = %f,val_acc = %f\n' % (cost_dev[i], acc_dev[i]))

        if err_dev[i] < best_err:
            best_err = err_dev[i]
            cprint('b', 'best acc')
            net.save(models_dir + '/theta_best.dat')

toc0 = time.time()
runtime_per_it = (toc0 - tic0) / float(nb_epochs)
cprint('r', '   average time: %f seconds\n' % runtime_per_it)

# net.save(models_dir+'/theta_last.dat')

# test
cost_test = 0
err_test = 0
acc_test = 0

net.load(models_dir + '/theta_best.dat')
cost, err, probs, acc, y_score = net.eval(X_test,Y_test)

cost_test += cost
err_test += err
acc_test += acc
nb_samples = len(X_test)

cost_test /= nb_samples
err_test /= nb_samples
acc_test /= nb_samples

print('  acc_test: %f' % (acc_test))

# AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i].detach().numpy().ravel())
    roc_auc[i] = auc(fpr[i], tpr[i])

print("auc of player0:%f" % roc_auc[0])
print("auc of player1:%f" % roc_auc[1])

fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.detach().numpy().ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print("auc of micro:%f" % roc_auc["micro"])

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='Bayes NNs roc of micro (auc = {0:0.2f})'
               ''.format(roc_auc["micro"]), color='green', linestyle='-', linewidth=1.5)
'''
plt.plot(fpr[0], tpr[0],
        label='Bayes NNs roc of player0(auc = {0:0.2f})'
                ''.format(roc_auc[0]),color='navy', linestyle=':', linewidth=1.5)

plt.plot(fpr[1], tpr[1],
        label='Bayes NNs roc of player1(auc = {0:0.2f})'
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
