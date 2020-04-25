import json
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

JSON_DATA_PATH = 'data.json'
EPOCHS = 50

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
    plt.title(f'Accuracy eval for {EPOCHS} epochs')
    plt.savefig(f'./pictures/cnn_{EPOCHS}_epochs_accuracy_plot.png')

    # create error subplot
    plt.clf()
    plt.figure(figsize=(16,8))
    plt.plot(history.history['loss'], label='train error')
    plt.plot(history.history['val_loss'], label='test error')
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='upper right')
    plt.title(f'Error eval for {EPOCHS} epochs')
    plt.savefig(f'./pictures/cnn_{EPOCHS}_epochs_error_plot.png')

def plot_confusion_matrix(con_mat_df):

    # plot confusion matrix
    plt.clf()
    plt.figure(figsize=(16,16))
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title(f'Confusion matrix for {EPOCHS} epochs')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'./pictures/cnn_{EPOCHS}_epochs_confusion_matrix.png')

def prepare_datasets(test_size, validation_size):

    # load data
    X, y, classes = load_data(JSON_DATA_PATH)

    # create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create the train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test, classes 

def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(
        keras.layers.Conv2D(
            32, (3, 3),
            activation='relu',
            input_shape=input_shape
        )
    )
    model.add(
        keras.layers.MaxPool2D(
            (3, 3),
            strides=(2, 2),
            padding='same'
        )
    )
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(
        keras.layers.Conv2D(
            32, (3, 3),
            activation='relu',
            input_shape=input_shape
        )
    )
    model.add(
        keras.layers.MaxPool2D(
            (3, 3),
            strides=(2, 2),
            padding='same'
        )
    )
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(
        keras.layers.Conv2D(
            32, (2, 2),
            activation='relu',
            input_shape=input_shape
        )
    )
    model.add(
        keras.layers.MaxPool2D(
            (2, 2),
            strides=(2, 2),
            padding='same'
        )
    )
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed to dense layer
    model.add(keras.layers.Flatten())
    model.add(
        keras.layers.Dense(
            64,
            activation='relu'
        )
    )
    model.add(
        keras.layers.Dropout(0.3)
    )

    # output layer
    model.add(
        keras.layers.Dense(
            10,
            activation='softmax'
        )
    )

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    print(f'Expected index:{y}')
    print(f'Predicted index:{predicted_index}')

if __name__ == '__main__':

    # Create train, validation, and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test, classes = prepare_datasets(test_size=0.25, validation_size=0.2)

    # Build CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # Compile CNN model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train CNN model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_validation, y_validation),
        batch_size=32,
        epochs=EPOCHS
    )

    # Evaluate CNN model on test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'\nAccuracy on test set:\t{test_accuracy}')
    print(f'Error on test set:\t{test_error}\n')

    # Make predictions on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)

    # Plot accuracy and error over epochs
    plot_history(history)

    # Plot confusion matrix
    y_pred = model.predict_classes(X_test)
    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)
    plot_confusion_matrix(con_mat_df)