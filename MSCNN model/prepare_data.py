import numpy as np
from sklearn.model_selection import train_test_split
import os

#get train sets,val sets and test sets
def getalldata(filepath):
    X, Y = [], []
    for file in os.listdir(filepath):
        filename = os.path.splitext(file)[0]
        #get labels
        winner = int(filename[-1:])
        #one-hot
        if winner == 0:
            Y.append([1, 0])
        elif winner == 1:
            Y.append([0, 1])
        else:
            raise ValueError("Invalid winner value!")

        x = np.load(os.path.join(filepath, file))
        X.append(x)
    X = np.array(X)  #(,8,8,38)
    Y = np.array(Y)  #(,2)

    #50% train sets，25% val sets，25% test set
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=1 / 2)
    X_test,X_val,Y_test,Y_val=train_test_split(X_val_test,Y_val_test,test_size=1/2)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)
    print(X_test.shape)
    print(Y_test.shape)
    return X_train, Y_train, X_val, Y_val,X_test,Y_test