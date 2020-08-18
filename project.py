import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50 , preprocess_input
from keras.models import Input , Model
from tensorflow.keras.preprocessing import image_dataset_from_directory





train_dir = r'chest_xray\train'
validation_dir = r'chest_xray\val'
test_dir = r'chest_xray\test'

BATCH_SIZE = 32
IMG_SIZE = (100, 100)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

test_dataset = image_dataset_from_directory(test_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

# =============================================================================
# class_names = train_dataset.class_names
# 
# 
# data_augmentation = tf.keras.Sequential([
#   tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
#   tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
# ])
# =============================================================================

input_shape = (100,100,3)
base_model = tf.keras.applications.ResNet50(weights='imagenet',input_shape = input_shape, include_top = False,
                      pooling = 'average')
base_model.trainable= False
preprocess = tf.keras.applications.resnet.preprocess_input
# =============================================================================
# fine_tune_at = 100
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable =  False
# =============================================================================

inputs = Input(shape=(100,100, 3))
x = preprocess(inputs)
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1000, activation = 'relu')(x)
x = Dense(1, activation = 'sigmoid')(x)
model = Model(base_model.input,x)


initial_epochs = 40


model.compile(Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


es = EarlyStopping(monitor='accuracy' , mode='max', patience = 5, restore_best_weights=True)


loss0, accuracy0 = model.evaluate(validation_dataset)

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,
                    callbacks=[es])


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(test_dataset)

print (score)