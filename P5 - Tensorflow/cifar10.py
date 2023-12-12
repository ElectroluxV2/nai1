"""
Author: pulkit12dhingra (https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/)
Author: Mateusz Budzisz

Tensorflow - classify images of airplane, automobile, bird, cat, deerdog, frog, horseship or truck

pip3 install urllib3==1.26.15 tensorflow matplotlib numpy
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

# Visualize data by plotting images
fig, ax = plt.subplots(5, 5)
k = 0

for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k], aspect='auto')
        k += 1

plt.show()

# Number of classes
K = len(set(y_train))

# Calculate total number of classes for output layer
print("number of classes:", K)

# Build the model using the functional API input layer
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)

# Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

# Last hidden layer i.e.. output layer
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# Model description
model.summary()

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

# Fit with data augmentation
# Note: if you run this AFTER calling
# the previous model.fit()
# it will CONTINUE training where it left off
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size  # // -> floor division

model.fit(train_generator, validation_data=(x_test, y_test), steps_per_epoch=steps_per_epoch, epochs=50)

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc', color='red')
plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
plt.legend()

# Label mapping
labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()

# Select the image from our test dataset
image_number = 0

# Display the image
plt.imshow(x_test[image_number])

# Load the image in an array
n = np.array(x_test[image_number])

# Reshape it
p = n.reshape(1, 32, 32, 3)

# Pass in the network for prediction and save the predicted label
predicted_label = labels[model.predict(p).argmax()]

# Load the original label
original_label = labels[y_test[image_number]]

# Display the result
print("Original label is {} and predicted label is {}".format(original_label, predicted_label))

# Save the model
model.save('cifar10.h5')
