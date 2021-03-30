from CNN import CNN
import tensorflow as tf
import matplotlib.pyplot as plt

cnn = CNN()

# Create Model - Swap arguments as per Q2
# Arg 1: Number of convolution layers
# Arg 2: Size of max pool
model = cnn.createModel(3, 2)

# Compile Model - Swap the different bits out here (mainly optimizer) as per Q2
optimizer = 'adam'
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
model = cnn.compileModel(model, optimizer, loss, metrics)

# Fit the model with training and validate with testing
history = cnn.fitModel(model, 10)

# Evalucate accuracy of model
test_acc = cnn.evaluateModel(model)
print(test_acc)

# Plot the data - Q3
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Q4?????
# Show the top 9 patches (regions that best activate neurons)
# https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
