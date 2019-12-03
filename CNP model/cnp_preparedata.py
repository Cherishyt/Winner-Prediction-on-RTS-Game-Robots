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
    return X, Y, X_test, Y_test

#get context sets and target sets
def split_c_t(x,y,batch_size):
    data_size=x.shape[0]
    total_batch = int(data_size / batch_size)
    context_x,context_y,target_x,target_y = [], [], [], []
    for i in range(total_batch):
        offset = (i * batch_size) % (data_size)
        batch_x = x[offset:(offset + batch_size), :]
        batch_y = y[offset:(offset + batch_size), :]
        batch_x_context,batch_x_target, batch_y_context, batch_y_target = train_test_split(batch_x, batch_y, test_size=1 / 2)
        context_x.append(batch_x_context)
        context_y.append(batch_y_context)
        target_x.append(batch_x_target)
        target_y.append(batch_y_target)
    context_x = np.array(context_x)
    context_y = np.array(context_y)
    target_x = np.array(target_x)
    target_y = np.array(target_y)
    print(context_x.shape)
    print(context_y.shape)
    print(target_x.shape)
    print(target_y.shape)
    return context_x,context_y,target_x,target_y