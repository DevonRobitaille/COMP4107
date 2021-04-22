# Some code referenced/taken from : https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/19-%20How%20to%20implement%20an%20RNN-LSTM%20for%20music%20genre%20classification/code/19-%20How%20to%20implement%20an%20RNN-LSTM%20for%20music%20genre%20classification.py

genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

import numpy as np
import sys
import tensorflow as tf
import librosa
import librosa.display
import os
import math
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.python.layers.normalization import BatchNormalization


# Building the model
def build_model(input_shape):
    # Set as sequential due to LSTM
    model = keras.Sequential()

    # Addition of LSTM layers, 1-2
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # Addition of the DENSE layers for dropping
    model.add(keras.layers.Dense(63, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # Addition of the output layer | Softmax allows for one output value per node in the output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


# load data
def load_data(dataset_path: str):
    features = np.get_features(dataset_path, ["gfcc", "mfcc"])
    return features

def plot_history(fitAccuracy, testAccuracy):
    # fig, axs = plt.subplots(1)

    # create accuracy sublpot
    plt.plot(fitAccuracy, label="train accuracy")
    plt.plot(testAccuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy eval")

    # create error sublpot
    # axs[1].plot(history.history["loss"], label="train error")
    # axs[1].plot(history.history["val_loss"], label="test error")
    # axs[1].set_ylabel("Error")
    # axs[1].set_xlabel("Epoch")
    # axs[1].legend(loc="upper right")
    # axs[1].set_title("Error eval")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    dataset_path = os.path.dirname(os.path.realpath(__file__))
    print(dataset_path)
    test_ratio = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.2
    num_training: int = math.floor(100 * (1 - test_ratio))

    sr_default = 1200

    dataset_path = dataset_path + "\genres";
    print(dataset_path)
    training_case_genre_folders = os.listdir(dataset_path)

    training_set = np.empty((10, num_training, sr_default, 20), dtype=np.float32)
    testing_set = np.empty((10, 100 - num_training, sr_default, 20), dtype=np.float32)

    print(training_case_genre_folders)
    for genre in range(len(training_case_genre_folders)):
        print(genre)
        # print(training_case_genre_folders[genre])
        music_path = os.path.join(dataset_path, training_case_genre_folders[genre])
        music_path = os.listdir(music_path)

        for music in range(len(music_path)):
            music_clip = os.path.join(dataset_path, training_case_genre_folders[genre], music_path[music])
            # print(music_clip)
            x, sr = librosa.load(music_clip)
            mfccs = librosa.feature.mfcc(x, sr=sr)
            mfccs = np.transpose(mfccs)

            if music < num_training:
                training_set[genre, music] = mfccs[:sr_default]
            else:
                testing_set[genre, music % num_training] = mfccs[:sr_default]

    # split the data into train and test sets
    # build the network architecture
    # compile network
    # train network

    input_shape = (testing_set.shape[2], testing_set.shape[3])
    print(input_shape)

    model = build_model(input_shape)

    model.summary()

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss="categorical_crossentropy",
                  # loss='mse',
                  metrics=["accuracy"])

    model.summary()
    print(training_set[:].shape)

    testing_y = np.zeros((10 * (100 - num_training), 10), dtype=np.int32)
    for i in range(10):
        for j in range((100 - num_training)):
            testing_y[i * (100 - num_training) + j, i] = 1

    print("Testing_y shape: ", testing_y.shape)

    testing_x = np.empty((0, testing_set.shape[2], testing_set.shape[3]), dtype=np.float32)
    for i in range(len(testing_set)):
        testing_x = np.concatenate((testing_x, testing_set[i]), axis=0)

    print("Testing_y shape: ", testing_x.shape)


    labels = np.zeros((10*num_training, 10), dtype=np.int32)
    for i in range (10):
        for j in range (num_training):
            labels[i*num_training + j, i] = 1


    print(labels.shape)


    training_x = np.empty((0, training_set.shape[2], training_set.shape[3]), dtype=np.float32)
    for i in range(len(training_set)):
        training_x = np.concatenate((training_x, training_set[i]), axis=0)

    print(training_x.shape)

    random_order = np.arange(training_x.shape[0])
    np.random.shuffle(random_order)
    labels = labels[random_order]
    training_x = training_x[random_order]

    validation_x = training_x[700:]
    training_x = training_x[:700]

    validation_y = labels[700:]
    labels = labels[:700]

    print("fitting model")
    # print(model.predict(training_x).shape)
    fitAccuracy = []
    testAccuracy = []
    for i in range(30):
        historyFit = model.fit(training_x, labels, validation_data=(validation_x, validation_y), batch_size=32)
        fitAccuracy.append(historyFit.history['accuracy'])
        historyTest = model.evaluate(testing_x, testing_y)
        for i in range(10):
            model.evaluate(testing_x[0+(20*i):20+(20*i)], testing_y[0+(20*i):20+(20*i)])
        testAccuracy.append(historyTest[1])
    plot_history(fitAccuracy, testAccuracy)
