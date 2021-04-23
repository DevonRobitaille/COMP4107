import numpy as np
import tensorflow as tf
import librosa

import tensorflow.keras as keras
import matplotlib.pyplot as plt

import os
import math

class LSTM_NN:
    # This function initializes the neural network with its necessary parameters
    # Inputs:
    #   sampling_rate: int - Used during the MFCC calculation for how many points of data should be returned
    #   data_path: str - the path to the correct working directory
    #   test_ratio: float - the percentage of data that should be used for the evaluation
    # Outputs:
    #   None
    def __init__(self, sampling_rate:int=1200, data_path:str=None, test_ratio:float=0.2, model_path=None):
        data_path = data_path + '\genres'
        training_case_genre_folders = os.listdir(data_path)

        self.num_training:int = math.floor(2 * (1 - test_ratio))
        self.sampling_rate = sampling_rate

        training_set = np.empty((10, self.num_training, self.sampling_rate, 20), dtype=np.float32)

        testing_set = []
        if model_path is None:
            testing_set = np.empty((10, 2 - self.num_training, self.sampling_rate, 20), dtype=np.float32)
        else:
            self.total_songs = 2
            testing_set = np.empty((10, 2, self.sampling_rate, 20), dtype=np.float32)
            

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
                # We transpose the mfcc because we want it to be (sampling rate, x) and not (x, sampling rate)
                mfccs = np.transpose(mfccs)

                if model_path is None:
                    # Create a training set and testing set
                    if music < self.num_training:
                        training_set[genre, music] = mfccs[:self.sampling_rate]
                    else:
                        testing_set[genre, music % self.num_training] = mfccs[:self.sampling_rate]

                else:
                    testing_set[genre, music] = mfccs[:self.sampling_rate]

        if model_path is None:
            # Create training_x, training_y
            self.training_y = np.zeros(((len(self.genres)) * self.num_training, 10), dtype=np.int32)
            for i in range((len(self.genres))):
                for j in range(self.num_training):
                    self.training_y[i * self.num_training + j, i] = 1

            self.training_x = np.empty((0, training_set.shape[2], training_set.shape[3]), dtype=np.float32)
            for i in range(len(training_set)):
                self.training_x = np.concatenate((self.training_x, training_set[i]), axis=0)

            # Randomize order of the training set
            random_order = np.arange(self.training_x.shape[0])
            np.random.shuffle(random_order)
            self.training_y = self.training_y[random_order]
            self.training_x = self.training_x[random_order]

        # Create testing_x, testing_y
        self.testing_x = np.empty((0, testing_set.shape[2], testing_set.shape[3]), dtype=np.float32)
        for i in range(len(testing_set)):
            self.testing_x = np.concatenate((self.testing_x, testing_set[i]), axis=0)
        self.total_songs = int(self.testing_x.shape[0]/10)
        print("total_songs: %d" % (self.total_songs))
        
        if model_path is None:
            self.testing_y = np.zeros(((len(self.genres)) * (2 - self.num_training), 10), dtype=np.int32)
        else :
            self.testing_y = np.zeros(((len(self.genres)) * self.total_songs, 10), dtype=np.int32)
        for i in range((len(self.genres))):
            for j in range(self.total_songs):
                self.testing_y[i * self.total_songs + j, i] = 1

    # This function creates a LSTM neural network model
    # Inputs:
    #   model_path: str - Path to a saved model (optional)
    def createModel(self, model_path=None): # Written By Graham
        if model_path is None:
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

        else:
            model = tf.keras.models.load_model(model_path)

            return model

    # This function takes as inputs a model which while be compiled with an optimiser, loss function, and metrics
    # Inputs: Model
    # Outputs: Compiled Model
    def compileModel(self, model): # Written By Graham
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])

        return model

    # This function trains the model and then uses the trained model to evualate its performance
    # Inputs:
    #   model
    #   epochs: int - how many epochs the model with be trained for
    # Outputs:
    #   fitAccuracy: Array - Contains how accurate the model during the training period
    #   testAccuracyAvg: Array - Contains how accurate the model was on average over the testing period
    #   testAccuracyIndividual: Array - Contains how accurate the model was for each individual genre for predicting during the testing period
    #   testTotalOccurences: Array - Contains how often the model choose a specific genre to the be prediction
    def fitModel(self, model, epochs:int=5): # Written By Alex
        fitAccuracy = []
        testAccuracyAvg = []
        testAccuracyIndividual = np.empty((epochs, 10), dtype=np.float32)
        testTotalOccurences = np.zeros((epochs, 10), dtype=np.int32)

        for i in range(epochs):
            historyFit = model.fit(self.training_x, self.training_y, batch_size=32) # validation_data=(validation_x, validation_y)
            fitAccuracy.append(historyFit.history['accuracy'])
            historyTestAvg = model.evaluate(self.testing_x, self.testing_y) # average peformance
            for j in range(10):
                testAccuracyIndividual[i, j] = model.evaluate(self.testing_x[self.total_songs*j:self.total_songs*(j+1)], self.testing_y[self.total_songs*j:self.total_songs*(j+1)])[1]

            for j in range (self.testing_x.shape[0]):
                data = self.testing_x[j:j+1]
                guess = model.predict(data)
                testTotalOccurences[i, np.argmax(guess)] += 1

            testAccuracyAvg.append(historyTestAvg[1])

            # Save the model
            # model_path = 'saved_model/lstm_model_' + str(i) + '_epochs'
            # model.save(model_path)

        return fitAccuracy, testAccuracyAvg, testAccuracyIndividual, testTotalOccurences

    def evaluateModel(self, model):
        historyTestAvg = model.evaluate(self.testing_x, self.testing_y) # average peformance

        testAccuracyIndividual = np.empty((1, 10), dtype=np.float32)
        for j in range(10):
            testAccuracyIndividual[0, j] = model.evaluate(self.testing_x[self.total_songs*j:self.total_songs*(j+1)], self.testing_y[self.total_songs*j:self.total_songs*(j+1)])[1]

        testTotalOccurences = np.zeros((1, 10), dtype=np.int32)
        for j in range (self.testing_x.shape[0]):
            data = self.testing_x[j:j+1]
            guess = model.predict(data)
            testTotalOccurences[0, np.argmax(guess)] += 1

        return historyTestAvg[1], testAccuracyIndividual, testTotalOccurences

    # Using the matplotlib library, graph the data collected during the fitModel funciton
    # Inputs:
    #   fitAccuracy: Array - Contains how accurate the model during the training period
    #   testAccuracyAvg: Array - Contains how accurate the model was on average over the testing period
    #   testAccuracyIndividual: Array - Contains how accurate the model was for each individual genre for predicting during the testing period
    #   testTotalOccurences: Array - Contains how often the model choose a specific genre to the be prediction
    def graphData(self, testAccuracyAvg, testAccuracyIndividual, testTotalOccurences, fitAccuracy=None): # Written By Alex
        # create accuracy sublpot for Avg Test versus Fit Train
        if not fitAccuracy is None:
            plt.plot(fitAccuracy, label="train accuracy")
        plt.plot(testAccuracyAvg, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend(loc="lower right")
        plt.title("Accuracy eval")

        # Show the plot
        plt.show()

        # accuracy of picking genres
        for i in range (testAccuracyIndividual.shape[1]):
            plt.plot(testAccuracyIndividual[:, i], label=self.genres[i])

        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend(loc="lower right")
        plt.title("Accuracy eval")

        # Show the plot
        plt.show()

        # tally by genre by epoch
        for i in range (testTotalOccurences.shape[1]):
            plt.plot(testTotalOccurences[:, i], label=self.genres[i])

        plt.ylabel("Tally")
        plt.xlabel("Epochs")
        plt.legend(loc="lower right")
        plt.title("Tally Recorded")

        # Show the plot
        plt.show()