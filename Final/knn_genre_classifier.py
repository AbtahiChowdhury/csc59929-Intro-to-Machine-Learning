import json
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

JSON_DATA_PATH = 'data.json'
NUM_OF_NEIGHBORS = 10

def plot_confusion_matrix(acclist, weights, neighbors):
    
    # Plot accuracy
    uniform_acclist = (acclist['uniform'])['acclist']
    distance_acclist = (acclist['distance'])['acclist']
    
    plt.clf()
    plt.figure(figsize=(16,16))
    plt.plot(neighbors,uniform_acclist,label='uniform')
    plt.plot(neighbors,distance_acclist,label='distance')
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f'Accuracy vs Number of Neighbors')
    plt.savefig('./pictures/knn_accuracy_plot.png')

    # Plot confusion matrix
    m = max(uniform_acclist)
    loc = [i for i,j in enumerate(uniform_acclist) if j==m]
    con_mat_df = ((acclist['uniform'])['con_mat'])[loc[0]]
    plt.clf()
    plt.figure(figsize=(16,16))
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title('Confusion matrix of KNN using uniform weights')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'./pictures/knn_uniform_confusion_matrix.png')
    
    m = max(distance_acclist)
    loc = [i for i,j in enumerate(distance_acclist) if j==m]
    con_mat_df = ((acclist['distance'])['con_mat'])[loc[0]]
    plt.clf()
    plt.figure(figsize=(16,16))
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title('Confusion matrix of KNN using distance weights')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('./pictures/knn_distance_confusion_matrix.png')


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
    #load data
    time = datetime.now().strftime('%H:%M:%S')
    print(f'[{time}] Reading data...\n')
    X, y, classes = load_data(JSON_DATA_PATH)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # weights to use
    weights = ['uniform', 'distance']

    # number of neighbors
    neighbors = [i+1 for i in range(NUM_OF_NEIGHBORS)]

    acclist = {
        'uniform': {
            'acclist': [],
            'con_mat': []
        },
        'distance': {
            'acclist': [],
            'con_mat': []
        }
    }

    for neighbor in neighbors:
        for weight in weights:

            # Create a new model
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Creating knn model using {weight} weights and {neighbor} neighbors...')
            model = KNeighborsClassifier(n_neighbors=neighbor, weights=weight)
            
            # Train the model
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Training model...')
            model.fit(X_train, y_train)

            # Print accuracy
            acc = model.score(X_test, y_test)
            (acclist[weight])['acclist'].append(acc)
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Accuracy of model: {acc}')

            # Make predictions using the trained model
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Predicting using trained model...')
            y_pred = model.predict(X_test)

            # Make confusion matrix
            time = datetime.now().strftime('%H:%M:%S')
            print(f'[{time}] Creating confusion matrix...')
            con_mat = confusion_matrix(y_test, y_pred)
            con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
            con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)
            (acclist[weight])['con_mat'].append(con_mat_df)

            print('\n')
    
    plot_confusion_matrix(acclist, weights, neighbors)