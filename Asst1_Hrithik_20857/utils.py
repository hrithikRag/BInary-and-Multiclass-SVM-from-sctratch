import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    X_train_new=[]
    X_test_new=[]
    for i in X_train.T:
        if i.max()-i.min()==0:
            normalized_col=i*0
        else:
            normalized_col = (2*(i-i.min())/(i.max()-i.min())) - 1
        X_train_new.append(normalized_col)
    
    for i in X_test.T:
        if i.max()-i.min()==0:
            normalized_col=i*0
        else:
            normalized_col = (2*(i-i.min())/(i.max()-i.min())) - 1
        X_test_new.append(normalized_col)
    
    return (np.array(X_train_new).T, np.array(X_test_new).T)
  


def plot_metrics(metrics) -> None:
    # plot and save the results
    k_arr=[]
    accuracy_arr=[]
    precision_arr=[]
    recall_arr=[]
    f1_score_arr=[]
    
    for i in range(7):
        k_arr.append(metrics[i][0])
        accuracy_arr.append(metrics[i][1])
        precision_arr.append(metrics[i][2])
        recall_arr.append(metrics[i][3])
        f1_score_arr.append(metrics[i][4])
        
    plt.plot(k_arr, accuracy_arr)
    plt.title("Accuracy K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.show()
    
    plt.plot(k_arr, precision_arr)
    plt.title("Precision vs K")
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.ylim(0,1.5)
    plt.show()
    
    plt.plot(k_arr, recall_arr)
    plt.title("Recall vs K")
    plt.xlabel("K")
    plt.ylabel("Recall")
    plt.ylim(0,1)
    plt.show()    
    
    plt.plot(k_arr, f1_score_arr)
    plt.title("f1_score vs K")
    plt.xlabel("K")
    plt.ylabel("f1_score")
    plt.ylim(0,1)
    plt.show()