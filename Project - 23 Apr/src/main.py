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

# load data
def load_data(dataset_path: str):
    features = get_features(dataset_path, ["gfcc", "mfcc"])

    return features

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    test_ratio   = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.2
    num_training:int = math.floor(100 * (1-test_ratio))

    sr_default = 1200

    training_case_genre_folders = os.listdir(dataset_path)

    training_set = np.empty((10, num_training, 20, sr_default), dtype=np.float32)
    testing_set = np.empty((10, 100-num_training, 20, sr_default), dtype=np.float32)

    for genre in range(len(training_case_genre_folders)):
        # print(training_case_genre_folders[genre])
        music_path = os.path.join(dataset_path, training_case_genre_folders[genre])
        music_path = os.listdir(music_path)

        for music in range(len(music_path)):
            music_clip = os.path.join(dataset_path, training_case_genre_folders[genre], music_path[music])
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