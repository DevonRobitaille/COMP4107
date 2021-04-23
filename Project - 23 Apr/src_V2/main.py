import numpy as np
import os
import sys

from LSTM_NN import LSTM_NN

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) >= 2 else None

    dataset_path = os.path.dirname(os.path.realpath(__file__))
    
    # Step 1 - Initialize the parameters of the network
    lstm = LSTM_NN(1200, dataset_path, 0.5, model_path)
    
    if model_path is None:
        # Step 2 - Build the model
        model = lstm.createModel()
        model.summary()

        # Step 3 - Compile the model
        model = lstm.compileModel(model)
        model.summary()

        # Step 4 - Fit the model
        fitAccuracy, testAccuracyAvg, testAccuracyIndividual, testTotalOccurences = lstm.fitModel(model, 30)

        # Step 5 - Graph different data points
        lstm.graphData(testAccuracyAvg, testAccuracyIndividual, testTotalOccurences, fitAccuracy)

    else:
        model_path = os.path.join(str(dataset_path), str(model_path))
        model_dir = os.listdir(model_path)
        print(model_path)

        testAccuracyAvg = []
        testAccuracyIndividual = np.empty((3, 10), dtype=np.float32)
        testTotalOccurences = np.zeros((3, 10), dtype=np.int32)

        for index in range(len(model_dir)):
            path = os.path.join(model_path, model_dir[index])

            # Step 2 - Build the model
            model = lstm.createModel(path)

            # Step 3 - Compile the model
            model = lstm.compileModel(model)

            # Step 4 - Fit the model
            tmp_testAccuracyAvg, tmp_testAccuracyIndividual, tmp_testTotalOccurences = lstm.evaluateModel(model)

            testAccuracyAvg.append(tmp_testAccuracyAvg)
            testAccuracyIndividual[index] = tmp_testAccuracyIndividual
            testTotalOccurences[index] = tmp_testTotalOccurences

        # Step 5 - Graph different data points
        lstm.graphData(testAccuracyAvg, testAccuracyIndividual, testTotalOccurences)
