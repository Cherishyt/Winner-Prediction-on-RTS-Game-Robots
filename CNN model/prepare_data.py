import numpy as np
from sklearn.model_selection import train_test_split
import os

#get train sets,val sets and test sets
def getalldata(filepath):
    T, W = [], []
    for file in os.listdir(filepath):
        filename = os.path.splitext(file)[0]
        #get labels
        winner = int(filename[-1:])
        #one-hot
        if winner == 0:
            W.append([1, 0])
        elif winner == 1:
            W.append([0, 1])
        else:
            raise ValueError("Invalid winner value!")
        t = np.load(os.path.join(filepath, file))
        T.append(t)
    T = np.array(T)  #(,8,8,39)
    W = np.array(W)  #(,2)

    #50% train sets，25% val sets，25% test set
    T_train, T_val_test,W_train, W_val_test = train_test_split(T, W, test_size=1 / 2)
    T_test,T_val,W_test,W_val=train_test_split(T_val_test,W_val_test,test_size=1/2)
    print(T_train.shape)
    print(W_train.shape)
    print(T_val.shape)
    print(W_val.shape)
    print(T_test.shape)
    print(W_test.shape)
    return T_train, W_train, T_val, W_val,T_test,W_test

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
