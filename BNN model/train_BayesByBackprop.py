__author__ = 'John'
import time
import torch.utils.data as Data
from torchvision import transforms, datasets
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from src.Bayes_By_Backprop.model import *
#from src.Bayes_By_Backprop_Local_Reparametrization.model import *
from sklearn.model_selection import train_test_split
import os
#the path of npy files
rootpath=r'E:\研\yutian\Datasets\randomsample_one_hot\npy'
models_dir = "BBP_model"
results_dir = "BBP_results"
mkdir(models_dir)
mkdir(results_dir)
# train config

batch_size = 50
nb_epochs = 100
lr=1e-4

# load data
T,W=[],[]
for file in os.listdir(rootpath):
    filename=os.path.splitext(file)[0]
    #get labels
    winner=int(filename[-1:])
    #one-hot
    if winner==0:
        W.append([1,0])
    elif winner==1:
        W.append([0,1])
    else:
        raise ValueError("Invalid winner value!")

    t=np.load(os.path.join(rootpath,file))
    T.append(t)

T = np.array(T)  #(,8,8,39)
W = np.array(W)  #(,2)
print(T.shape)
print(W.shape)
#50% train sets，25% val sets，25% test set
T_train, T_val_test, W_train, W_val_test = train_test_split(T, W, test_size=1 / 2)
T_test, T_val, W_test, W_val = train_test_split(T_val_test, W_val_test, test_size=1 / 2)
print(T_train.shape)
print(W_train.shape)
print(T_val.shape)
print(W_val.shape)
print(T_test.shape)
print(W_test.shape)

NTrainPoints=T_train.shape[0]
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


#model
net = BBP_Bayes_Net(lr=lr, channels_in=39, side_in=8, cuda=use_cuda, classes=2, batch_size=batch_size,
                    Nbatches=(NTrainPoints/ batch_size), nhid=1200, prior_instance=laplace_prior(mu=0, b=0.1))
                    #prior_instance=isotropic_gauss_prior(mu=0, sigma=0.1))


kl_cost_train = np.zeros(nb_epochs)
pred_cost_train = np.zeros(nb_epochs)
err_train = np.zeros(nb_epochs)
acc_train=np.zeros(nb_epochs)

cost_dev = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)
acc_dev = np.zeros(nb_epochs)
best_err = np.inf

nb_its_dev = 1

tic0 = time.time()
for i in range(0, nb_epochs):
    # We draw more samples on the first epoch in order to ensure convergence
    if i == 0:
        ELBO_samples = 10
    else:
        ELBO_samples = 3

    net.set_mode_train(True)
    tic = time.time()
    nb_samples = 0

    #train
    for batch_T, batch_W in minibatches(T_train, W_train, batch_size, shuffle=True):
        cost_dkl, cost_pred, err,acc = net.fit(batch_T, batch_W, samples=ELBO_samples)

        err_train[i] += err
        kl_cost_train[i] += cost_dkl
        pred_cost_train[i] += cost_pred
        acc_train[i]+=acc
        nb_samples += len(batch_T)

    kl_cost_train[i] /= nb_samples  # Normalise by number of samples in order to get comparable number to the -log like
    pred_cost_train[i] /= nb_samples
    err_train[i] /= nb_samples
    acc_train[i]/= nb_samples

    toc = time.time()
    net.epoch = i
    # ---- print
    print("it %d/%d, Jtr_KL = %f, Jtr_pred = %f, err = %f, acc = %f," % (
    i, nb_epochs, kl_cost_train[i], pred_cost_train[i], err_train[i],acc_train[i]))

    # ---- dev
    #val
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        nb_samples = 0
        
        cost, err, probs,acc,out = net.eval(T_val, W_val)  # This takes the expected weights to save time, not proper inference

        cost_dev[i] += cost
        err_dev[i] += err
        acc_dev[i] += acc
        nb_samples += len(T_val)

        cost_dev[i] /= nb_samples
        err_dev[i] /= nb_samples
        acc_dev[i] /= nb_samples


        if err_dev[i] < best_err:
            best_err = err_dev[i]
            print('best_err_dev: %f' % (err_dev[i]))
            print('best_acc_dev: %f' % (acc_dev[i]))
            best_out=out
            net.save(models_dir + '/theta_best.dat')

toc0 = time.time()
runtime_per_it = (toc0 - tic0) / float(nb_epochs)
cprint('r', '   average time: %f seconds\n' % runtime_per_it)
net.save(models_dir + '/theta_last.dat')


# results
cprint('c', '\nRESULTS:')
nb_parameters = net.get_nb_parameters()
best_cost_dev = np.min(cost_dev)
best_cost_train = np.min(pred_cost_train)
err_dev_min = err_dev[::nb_its_dev].min()

print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
print('  err_dev: %f' % (err_dev_min))
print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
print('  time_per_it: %fs\n' % (runtime_per_it))


#test
cost_test=0
err_test=0
acc_test=0
nb_samples=0
net.load(models_dir + '/theta_best.dat')
cost, err, probs, acc,y_score = net.eval(T_test, W_test)  # This takes the expected weights to save time, not proper inference

cost_test += cost
err_test += err
acc_test += acc
nb_samples += len(T_test)

cost_test /= nb_samples
err_test /= nb_samples
acc_test /= nb_samples

print('  acc_test: %f' % (acc_test))

#AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i],tpr[i],_ = roc_curve(W_test[:,i], y_score[:,i].detach().numpy().ravel())
    roc_auc[i] = auc(fpr[i],tpr[i])

print("auc of player0:%f"%roc_auc[0])
print("auc of player1:%f"%roc_auc[1])

fpr["micro"], tpr["micro"], _ = roc_curve(W_test.ravel(), y_score.detach().numpy().ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print("auc of micro:%f" % roc_auc["micro"])


plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='Bayes NNs roc of micro (auc = {0:0.2f})'
               ''.format(roc_auc["micro"]), color='green', linestyle='-', linewidth=1.5)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curves and AUC values')
plt.legend(loc="lower right")
plt.show()
