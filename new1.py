import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = [224, 224]
WIDTH = 151
HEIGHT = 136
BATCH_SIZE = 16  
num_epochs = 50

train_path = 'C:/Users/UDAY SAI/OneDrive/Desktop/og/Dataset/Train'
test_path = 'C:/Users/UDAY SAI/OneDrive/Desktop/og/Dataset/Test'

vgg19 = VGG19(input_shape=image_size+[3], weights="imagenet", include_top=False)

for layers in vgg19.layers:
    layers.trainable=False

x = Flatten()(vgg19.output)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=vgg19.input, outputs=prediction)
model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                target_size=(224, 224),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224, 224),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical')

steps_per_epoch = len(training_set)
try:
    history = model.fit(training_set, validation_data=test_set, epochs=num_epochs)
except Exception as e:
    print(f"An error occurred during training: {e}")

# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.legend()
# plt.show()
# plt.savefig('LossVal_loss.png')

# plt.plot(history.history['accuracy'], label='train acc')
# plt.plot(history.history['val_accuracy'], label='val acc')
# plt.legend()
# plt.show()
# plt.savefig('AccVal_acc.png')

model.save('malaria_model.h5')


