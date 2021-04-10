from CNN import CNN
import tensorflow as tf
import matplotlib.pyplot as plt
from math import floor

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

# Q4
labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

best_9_maps, f_maps = cnn.getFeatureMap(9)
plt.figure()
f, axarr = plt.subplots(6, 3)
for i in range (9):
    t_img = cnn.test_images[ best_9_maps[i][1] ]
    t_img_gray = cnn._rgb2gray(t_img)
    
    # Original Image
    axarr[floor(i/3)*2, (i%3)].imshow(t_img_gray)
    axarr[floor(i/3)*2, (i%3)].set_title('Original ' + labels[cnn.test_labels[ best_9_maps[i][1] ][0]])
    
    # Feature Map
    axarr[floor(i/3)*2+1, (i%3)].imshow(f_maps[best_9_maps[i][1], :, :, best_9_maps[i][2]])
    axarr[floor(i/3)*2+1, (i%3)].set_title('Feature Map ' + labels[cnn.test_labels[ best_9_maps[i][1] ][0]])

f.subplots_adjust(hspace=2, wspace=1)
plt.show()
