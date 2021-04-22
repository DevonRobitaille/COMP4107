import numpy as np
import os
from LSTM_NN import LSTM_NN

if __name__ == "__main__":
    dataset_path = os.path.dirname(os.path.realpath(__file__))
    
    # Step 1 - Initialize the parameters of the network
    lstm = LSTM_NN(1200, dataset_path, 0.2)
    print(lstm.training_y.shape)
    print(lstm.training_x.shape)
    print(lstm.testing_y.shape)
    print(lstm.testing_x.shape)

    # Step 2 - Build the model
    model = lstm.createModel()
    model.summary()

    # Step 3 - Compile the model
    model = lstm.compileModel(model)
    model.summary()

    # Step 4 - Fit the model
    fitAccuracy, testAccuracy = lstm.fitModel(model)

    # Step 5 - Predict using the model

    # Step 6 - Graph different data points
    lstm.graphData(fitAccuracy, testAccuracy)
