#Some code referenced/taken from : https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/19-%20How%20to%20implement%20an%20RNN-LSTM%20for%20music%20genre%20classification/code/19-%20How%20to%20implement%20an%20RNN-LSTM%20for%20music%20genre%20classification.py 



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


#Building the model
def build_model(input_shape):
    #Set as sequential due to LSTM
    model = keras.Sequential()
    
    #Addition of LSTM layers, 1-2 
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))
    
    #Addition of the DENSE layers for dropping
    model.add(keras.layers.Dense(63, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    #Addition of the output layer | Softmax allows for one output value per node in the output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model


# load data
def load_data(dataset_path: str):
    features = np.get_features(dataset_path, ["gfcc", "mfcc"])
    return features

if __name__ == "__main__":
    dataset_path = os.path.dirname(os.path.realpath(__file__))
    print(dataset_path)
    test_ratio   = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.2
    num_training:int = math.floor(100 * (1-test_ratio))

    sr_default = 1200
    
    dataset_path = dataset_path + "\genres";
    print(dataset_path)
    training_case_genre_folders = os.listdir(dataset_path)

    training_set = np.empty((10, num_training, 20, sr_default), dtype=np.float32)
    testing_set = np.empty((10, 100-num_training, 20, sr_default), dtype=np.float32)

    for genre in range(len(training_case_genre_folders)):
        #print(training_case_genre_folders[genre])
        music_path = os.path.join(dataset_path, training_case_genre_folders[genre])
        music_path = os.listdir(music_path)

        for music in range(len(music_path)):
            music_clip = os.path.join(dataset_path, training_case_genre_folders[genre], music_path[music])
            #print(music_clip)
            x, sr = librosa.load(music_clip)
            mfccs = librosa.feature.mfcc(x, sr=sr)

            if music < num_training:
                training_set[genre, music] = mfccs[:, :sr_default]
            else:
                testing_set[genre, music%num_training] = mfccs[:, :sr_default]


    # split the data into train and test sets
    # build the network architecture
    # compile network
    # train network
    
    input_shape=(testing_set.shape[2], testing_set.shape[3])
    
    model = build_model(input_shape)
    
    model.summary()
    
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                 loss="categorical_crossentropy",
                 metrics=["accuracy"])
    
    model.summary()
    print(training_set[:].shape)
    
    labels = np.empty((10, num_training), dtype=np.int32)
    for i in range(10):
        data = np.empty((num_training), dtype=np.int32)
        data.fill(i)
        #labels = np.concatenate((labels, data))
        labels[i] = data
        
    labels = labels.flatten()
    
    print(labels.shape)
    
    training_x = np.empty((0, training_set.shape[2], training_set.shape[3]), dtype=np.float32)
    for i in range(len(training_set)):
        training_x = np.concatenate((training_x, training_set[i]), axis=0)

    print(training_x.shape)
    
    #history = model.fit(training_set[:], testing_set[])


    

def plot_history(history):
    fig, axs = plt.subplots(2)
    
    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    #Show the plot
    plt.show()    
    
