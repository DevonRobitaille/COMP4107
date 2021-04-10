# Load dependencies
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import math

class CNN:
    def __init__(self):
        # load data set
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0

        print(self.train_images.shape[0])
        print(self.train_images.shape[1])

    def createModel(self, num_conv_layers: int, size_pool: int):
        model = models.Sequential()

        # Create the convolution base
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

        for layer in range (num_conv_layers - 1):
            model.add(layers.MaxPooling2D((size_pool, size_pool)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))      

        # Add Dense layers on top 
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))

        return model

    def compileModel(self, model, optimizer: str, loss, metrics):
        model.compile(optimizer = optimizer,
                      loss = loss,
                      metrics = metrics
                    )

        return model

    def fitModel(self, model, epochs: int):
        history = model.fit(self.train_images, self.train_labels, epochs=epochs, validation_data=(self.test_images, self.test_labels))

        return history

    def evaluateModel(self, model):
        patches = tf.image.extract_patches(
                                            images=self.test_images,
                                            sizes=[1, 3, 3, 1],
                                            strides=[1, 5, 5, 1],
                                            rates=[1, 1, 1, 1],
                                            padding='VALID'
                                        )

        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels, verbose=2)
        return test_acc

    def getFeatureMap(self, num_patches):
        # Create Model
        input_shape = (32, 32, 3)
        inputs = tf.keras.Input(input_shape)
        outputs = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=input_shape)(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        # Compile Model
        model.compile(optimizer = 'adam',
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics = ['accuracy']
                    )

        # # Get feature map through predict
        feature_maps = model.predict(self.test_images) # Verify that this works!

        print(feature_maps.shape)
        print(self.test_images.shape)

        # Loop over all of the test_images
        dtype = [('dist', float), ('test_img', int), ('f_map', int)]
        best_9_maps = np.zeros((num_patches), dtype=dtype)

        for t_img in range (feature_maps.shape[0]):

            dist_from_norm = np.zeros((feature_maps.shape[3]), dtype=dtype)
            t_img_gray = self._rgb2gray(self.test_images[t_img])   
 
            for f_map in range (feature_maps.shape[3]):
                dist_from_norm[f_map] = (LA.norm(t_img_gray - feature_maps[t_img, :, :, f_map]), t_img, f_map)

            dist_from_norm = np.sort(dist_from_norm, axis=None, order='dist')[:num_patches]

            
            bool_arr = np.all((best_9_maps == best_9_maps[0]))
            if (bool_arr):
                best_9_maps = dist_from_norm
            else:
                # pick the 9 best nine
                best_9_maps = np.concatenate((best_9_maps, dist_from_norm), axis=None)
                best_9_maps = np.sort(best_9_maps, axis=None, order='dist')[:num_patches]
    
        return best_9_maps, feature_maps

    def _rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])