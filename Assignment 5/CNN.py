# Load dependencies
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

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

        #  Add Dense layers on top 
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
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels, verbose=2)
        return test_acc