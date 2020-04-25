import json
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

JSON_DATA_PATH = 'data.json'
EPOCHS = 100

def load_data(dataset_path):

    # load data from json file
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
    
    # convert list into numpy arrays
    X = np.array(data['mfcc'])
    y = np.array(data['labels'])
    classes = np.array(data['mapping'])

    return X, y, classes

def plot_history(history):

    # create accuracy subplot
    plt.clf()
    plt.figure(figsize=(16,8))
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='test accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.title(f'Accuracy eval for {EPOCHS} epochs with dropout and L2 regularization')
    plt.savefig(f'./pictures/nn_{EPOCHS}_epochs_accuracy_plot_fix_overfitting.png')

    # create error subplot
    plt.clf()
    plt.figure(figsize=(16,8))
    plt.plot(history.history['loss'], label='train error')
    plt.plot(history.history['val_loss'], label='test error')
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='upper right')
    plt.title(f'Error eval for {EPOCHS} epochs with dropout and L2 regularization')
    plt.savefig(f'./pictures/nn_{EPOCHS}_epochs_error_plot_fix_overfitting.png')

def plot_confusion_matrix(con_mat_df):

    # plot confusion matrix
    plt.clf()
    plt.figure(figsize=(16,16))
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title(f'Confusion matrix for {EPOCHS} epochs')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'./pictures/nn_{EPOCHS}_epochs_confusion_matrix_fix_overfitting.png')


if __name__ == '__main__':
    
    #load data
    X, y, classes = load_data(JSON_DATA_PATH)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build the network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st lidden layer
        keras.layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        keras.layers.Dropout(0.3),

        # 2nd lidden layer
        keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        keras.layers.Dropout(0.3),

        # 3rd lidden layer
        keras.layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # train network
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=32
    )

    # plot accuracy and error over the epochs
    plot_history(history)

    # plot confusion matrix
    y_pred = model.predict_classes(X_test)
    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)
    plot_confusion_matrix(con_mat_df)

    

