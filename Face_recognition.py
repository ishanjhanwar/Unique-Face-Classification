from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os

# re-size all the images to this
IMAGE_SIZE = [64, 64]

train_path = os.path.join('./dataset/Train')
test_path = os.path.join('./dataset/Test')

INPUT_SIZE = IMAGE_SIZE + [3]
model = Sequential([
    Input(shape=INPUT_SIZE),
    Conv2D(8, 7, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    MaxPooling2D(),
    Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    Dropout(rate=0.3),
    Conv2D(8, 7, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    Flatten(),
    Dense(5, activation='softmax')
])

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=10,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')
print(training_set.class_indices)


# fit the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5
)

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss.jpg')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc.jpg')

model.save('facefeatures_model.h5')
