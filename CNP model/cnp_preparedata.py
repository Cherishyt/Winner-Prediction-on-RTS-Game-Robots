import numpy as np
from sklearn.model_selection import train_test_split
import os

#get train sets and test sets
def get_train_test_data(filepath):
    T, W = [], []
    for file in os.listdir(filepath):
        filename = os.path.splitext(file)[0]
        #get labels
        winner = int(filename[-1:])
        #one-hot
        if winner == 0:
            W.append([1,0])
        elif winner == 1:
            W.append([0,1])
        else:
            raise ValueError("Invalid winner value!")

        t = np.load(os.path.join(filepath, file))
        t=t.reshape([-1])
        T.append(t)
    T = np.array(T)
    W = np.array(W)

    T_train, T_test, W_train, W_test = train_test_split(T, W, test_size=1 / 3)
    print(T_train.shape)
    print(W_train.shape)
    print(T_test.shape)
    print(W_test.shape)
    return T_train, W_train, T_test, W_test

#get context sets and target sets
def split_c_t(T,W,batch_size,shuffle=False):
    data_size=T.shape[0]
    i=0
    if shuffle:
        indices=np.arange(data_size)
        np.random.shuffle(indices)
    for start_idx in range(0,data_size-batch_size+1,batch_size):
        if shuffle:
            excerpt=indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        batch_T=np.expand_dims(T[excerpt],axis=1)
        batch_W=np.expand_dims(W[excerpt],axis=1)
        if i==0:
            S_T=batch_T
            S_W=batch_W
        else:
            S_T=np.concatenate((S_T,batch_T),axis=1)
            S_W=np.concatenate((S_W,batch_W),axis=1)
        i+=1
    num_observations=S_T.shape[1]
    num_target=int(num_observations/2)
    c_T=S_T[: ,:num_target,:]
    c_W = S_W[: ,:num_target, :]
    t_T = S_T[: , num_target: , :]
    t_W = S_W[: , num_target: , :]

    return c_T,c_W,t_T,t_W
