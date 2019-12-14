import numpy as np
from sklearn.model_selection import train_test_split
import os

#get train sets and test sets
def get_train_test_data(filepath):
    X, Y = [], []
    for file in os.listdir(filepath):
        filename = os.path.splitext(file)[0]
        #get labels
        winner = int(filename[-1:])
        #one-hot
        if winner == 0:
            Y.append([1,0])
        elif winner == 1:
            Y.append([0,1])
        else:
            raise ValueError("Invalid winner value!")

        x = np.load(os.path.join(filepath, file))
        x=x.reshape([-1])
        X.append(x)
    X = np.array(X)
    Y = np.array(Y)

    #50% train sets，50% test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 2)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    return X_train, Y_train, X_test, Y_test

#get context sets and target sets
def split_c_t(x,y,batch_size):
    data_size=x.shape[0]
    total_batch = int(data_size / batch_size)
    #sub-sets S_x,S_y
    for i in range(total_batch):
        offset = (i * batch_size) % (data_size)
        batch_x = x[offset:(offset + batch_size), :]
        batch_y = y[offset:(offset + batch_size), :]
        batch_x = np.expand_dims(batch_x,axis=1)
        batch_y = np.expand_dims(batch_y, axis=1)
        if i==0:
            S_x=batch_x
            S_y=batch_y
        else:
            S_x=np.concatenate((S_x,batch_x),axis=1)
            S_y=np.concatenate((S_y,batch_y),axis=1)
    print(S_x.shape)
    print(S_y.shape)
    num_observations=S_x.shape[1]
    num_target=int(num_observations/2)
    print(num_target)
    context_x=S_x[: ,:num_target,:]
    context_y = S_y[: ,:num_target, :]
    target_x = S_x[: , num_target: , :]
    target_y = S_y[: , num_target: , :]

    print(context_x.shape)
    print(context_y.shape)
    print(target_x.shape)
    print(target_y.shape)
    return S_x,S_y,target_x,target_y
