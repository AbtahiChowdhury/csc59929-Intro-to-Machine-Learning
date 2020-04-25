import json
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

JSON_DATA_PATH = 'data.json'
NUM_REGULARIZATIONS = 10

def plot_confusion_matrix(acclist, kernels, c_vals):

    # Plot accuracy
    linear_acclist = (acclist['linear'])['acclist']
    poly_acclist = (acclist['poly'])['acclist']
    sigmoid_acclist = (acclist['sigmoid'])['acclist']
    rbf_acclist = (acclist['rbf'])['acclist']
    
    plt.clf()
    plt.figure(figsize=(16,16))
    plt.plot(c_vals,linear_acclist,label='linear')
    plt.plot(c_vals,poly_acclist,label='poly')
    plt.plot(c_vals,sigmoid_acclist,label='sigmoid')
    plt.plot(c_vals,rbf_acclist,label='rbf')
    plt.xlabel("Regularization Factor")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f'Accuracy vs Regularization Factor')
    plt.savefig('./pictures/svm_accuracy_plot.png')

    # Plot confusion matrix
    m = max(linear_acclist)
    loc = [i for i,j in enumerate(linear_acclist) if j==m]
    con_mat_df = ((acclist['linear'])['con_mat'])[loc[0]]
    plt.clf()
    plt.figure(figsize=(16,16))
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title('Confusion matrix of SVM using linear kernel')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'./pictures/svm_linear_confusion_matrix.png')
    
    m = max(poly_acclist)
    loc = [i for i,j in enumerate(poly_acclist) if j==m]
    con_mat_df = ((acclist['poly'])['con_mat'])[loc[0]]
    plt.clf()
    plt.figure(figsize=(16,16))
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title('Confusion matrix of SVM using poly kernel')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('./pictures/svm_poly_confusion_matrix.png')

    m = max(sigmoid_acclist)
    loc = [i for i,j in enumerate(sigmoid_acclist) if j==m]
    con_mat_df = ((acclist['sigmoid'])['con_mat'])[loc[0]]
    plt.clf()
    plt.figure(figsize=(16,16))
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title('Confusion matrix of SVM using sigmoid kernel')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('./pictures/svm_sigmoid_confusion_matrix.png')

    m = max(rbf_acclist)
    loc = [i for i,j in enumerate(rbf_acclist) if j==m]
    con_mat_df = ((acclist['rbf'])['con_mat'])[loc[0]]
    plt.clf()
    plt.figure(figsize=(16,16))
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title('Confusion matrix of SVM using rbf kernel')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('./pictures/svm_rbf_confusion_matrix.png')


def load_data(dataset_path):

    # load data from json file
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
    
    # convert list into numpy arrays
    X = np.array(data['mfcc'])
    y = np.array(data['labels'])
    classes = np.array(data['mapping'])

    return X, y, classes


if __name__ == '__main__':

    # Regularization factors
    c_vals = [i for i in range(10, (NUM_REGULARIZATIONS*10)+10, 10)]
    
    # Kernels
    kernels = ['linear','poly','sigmoid','rbf']
    acclist = {
        'linear': {
            'acclist': [],
            'con_mat': []
        },
        'poly': {
            'acclist': [],
            'con_mat': []
        },
        'sigmoid': {
            'acclist': [],
            'con_mat': []
        },
        'rbf': {
            'acclist': [],
            'con_mat': []
        }
    }

    #load data
    time = datetime.now().strftime('%H:%M:%S')
    print(f'[{time}] Reading data...\n')
    X, y, classes = load_data(JSON_DATA_PATH)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    for c in c_vals:
        for kernel in kernels:

            # Create the model
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Creating svm model using {kernel} kernel and regularization factor {c}...')
            svm_linear = SVC(C=c, kernel=kernel)

            # Train the model
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Training model...')
            svm_linear.fit(X_train, y_train)

            # Print accuracy
            acc = svm_linear.score(X_test, y_test)
            (acclist[kernel])['acclist'].append(acc)
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Accuracy for {kernel} svm: {acc}')

            # Pridicte using trained model
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Predicting using trained model...')
            y_pred = svm_linear.predict(X_test)

            # Make confusion matrix
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Creating confusion matrix...')
            con_mat = confusion_matrix(y_test, y_pred)
            con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
            con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)
            (acclist[kernel])['con_mat'].append(con_mat_df)

            print('\n')
    
    plot_confusion_matrix(acclist, kernels, c_vals)
