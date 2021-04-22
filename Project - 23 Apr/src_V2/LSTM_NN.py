import numpy as np
import tensorflow as tf
import librosa

import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.python.layers.normalization import BatchNormalization

import os
import math

class LSTM_NN:
    def __init__(self, sampling_rate:int=1200, data_path:str=None, test_ratio:float=0.2):
        data_path = data_path + '\genres'
        training_case_genre_folders = os.listdir(data_path)

        self.num_training:int = math.floor(100 * (1 - test_ratio))
        self.sampling_rate = sampling_rate

        training_set = np.empty((10, self.num_training, self.sampling_rate, 20), dtype=np.float32)
        testing_set = np.empty((10, 100 - self.num_training, self.sampling_rate, 20), dtype=np.float32)

        self.genres = []

        # Build training set and testing set
        for genre in range(len(training_case_genre_folders)): # Written By Devon
            # Create the different possible outputs
            self.genres.append(training_case_genre_folders[genre])

            music_path = os.path.join(data_path, training_case_genre_folders[genre])
            music_path = os.listdir(music_path)

            for music in range(len(music_path)):
                # Pull data from music files that will be used to training the neural network (MFCC)
                music_clip = os.path.join(data_path, training_case_genre_folders[genre], music_path[music])
                x, sr = librosa.load(music_clip)
                mfccs = librosa.feature.mfcc(x, sr=sr)
                mfccs = np.transpose(mfccs)

                # Create a training set and testing set
                if music < self.num_training:
                    training_set[genre, music] = mfccs[:self.sampling_rate]
                else:
                    testing_set[genre, music % self.num_training] = mfccs[:self.sampling_rate]

        print(self.genres)

        # Create training_x, training_y
        self.training_y = np.zeros(((len(self.genres)) * self.num_training, 10), dtype=np.int32)
        for i in range((len(self.genres))):
            for j in range(self.num_training):
                self.training_y[i * self.num_training + j, i] = 1

        self.training_x = np.empty((0, training_set.shape[2], training_set.shape[3]), dtype=np.float32)
        for i in range(len(training_set)):
            self.training_x = np.concatenate((self.training_x, training_set[i]), axis=0)

        # Randomize order
        random_order = np.arange(self.training_x.shape[0])
        np.random.shuffle(random_order)
        self.training_y = self.training_y[random_order]
        self.training_x = self.training_x[random_order]

        # Create testing_x, testing_y
        self.testing_y = np.zeros(((len(self.genres)) * (100 - self.num_training), 10), dtype=np.int32)
        for i in range((len(self.genres))):
            for j in range((100 - self.num_training)):
                self.testing_y[i * (100 - self.num_training) + j, i] = 1

        self.testing_x = np.empty((0, testing_set.shape[2], testing_set.shape[3]), dtype=np.float32)
        for i in range(len(testing_set)):
            self.testing_x = np.concatenate((self.testing_x, testing_set[i]), axis=0)

        # Create validation data
        # validation_x = training_x[700:]
        # training_x = training_x[:700]

        # validation_y = labels[700:]
        # labels = labels[:700]


    def createModel(self): # Written By Graham
        # Set as sequential due to LSTM
        model = keras.Sequential()

        # Addition of LSTM layers, 1-2
        model.add(keras.layers.LSTM(64, input_shape=(self.training_x.shape[1], self.training_x.shape[2]), return_sequences=True))
        model.add(keras.layers.LSTM(64))

        # Addition of the DENSE layers for dropping
        model.add(keras.layers.Dense(63, activation='relu'))
        model.add(keras.layers.Dropout(0.3))

        # Addition of the output layer | Softmax allows for one output value per node in the output layer
        model.add(keras.layers.Dense(10, activation='softmax'))

        return model

    def compileModel(self, model):
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                    loss="categorical_crossentropy",
                    # loss='mse',
                    metrics=["accuracy"])

        return model

    def fitModel(self, model, epochs:int=5): # Written By Alex
        fitAccuracy = []
        testAccuracyAvg = []
        testAccuracyIndividual = np.empty((epochs, 10), dtype=np.float32)
        for i in range(epochs):
            historyFit = model.fit(self.training_x, self.training_y, batch_size=32) # validation_data=(validation_x, validation_y)
            fitAccuracy.append(historyFit.history['accuracy'])
            historyTestAvg = model.evaluate(self.testing_x, self.testing_y) # average peformance
            for j in range(10):
                testAccuracyIndividual[i, j] = model.evaluate(self.testing_x[20*j:20*(j+1)], self.testing_y[20*j:20*(j+1)])[1]
            testAccuracyAvg.append(historyTestAvg[1])

        return fitAccuracy, testAccuracyAvg, testAccuracyIndividual

    def evaluateModel(self, model):
        return test_acc

    def graphData(self, fitAccuracy, testAccuracyAvg, testAccuracyIndividual): # Written By Alex
        # create accuracy sublpot
        plt.plot(fitAccuracy, label="train accuracy")
        plt.plot(testAccuracyAvg, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend(loc="lower right")
        plt.title("Accuracy eval")

        # Show the plot
        plt.show()

        for i in range (testAccuracyIndividual.shape[1]):
            plt.plot(testAccuracyIndividual[:, i], label=self.genres[i])

        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend(loc="lower right")
        plt.title("Accuracy eval")

        # Show the plot
        plt.show()