# Written by Satish Chandra

# Import the modules
import numpy as np
import tensorflow as tf  # or import torch for PyTorch
from sklearn.model_selection import train_test_split

# Load the MNIST dataset,
# Which consists of 28x28 pixel grayscale images of handwritten digits.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten the images and normalize the pixel values to the range [0, 1].
X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

# Split the data into training and testing sets.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create a simple neural network using a framework like TensorFlow or PyTorch.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with an appropriate loss function, optimizer, and metrics.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training data.
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Evaluate the model on the test set to see how well it generalizes.
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Use the trained model to make predictions on new data.
predictions = model.predict(X_test[:5])

# Optionally, visualize the model's predictions and check how well it performs on sample images.

import matplotlib.pyplot as plt

for i in range(5):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}')
    plt.show()
