import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory

test_dir = r'chest_xray\test'
train_dir = r'chest_xray\train'
val_dir = r'chest_xray\val'

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

test_dataset = image_dataset_from_directory(test_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

val_dataset = image_dataset_from_directory(val_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)


class_names = train_dataset.class_names



preprocess = tf.keras.applications.resnet.preprocess_input

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

img_shape = IMG_SIZE + (3,)
base_model = tf.keras.applications.ResNet50(weights='imagenet',pooling = 'average',input_shape = img_shape, include_top = False)
#base_model.trainable = False #freeze all, add fine tuning later

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False


#rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

dense_layer = tf.keras.layers.Dense(256, activation = 'relu')
prediction_layer = tf.keras.layers.Dense(1)


inputs = tf.keras.Input(shape=img_shape)
x=preprocess(inputs)
x = data_augmentation(x)
x = base_model(x)
x = global_average_layer(x)
#x = tf.keras.layers.Dropout(0.25)(x)
#x= tf.keras.layers.BatchNormalization(x)
x = dense_layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

initial_epochs = 100
es = EarlyStopping(monitor='val_loss' , mode='min', patience = 4, restore_best_weights=True)
loss0, accuracy0 = model.evaluate(val_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=val_dataset,
                    callbacks = [es])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

"""
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
"""
score = model.evaluate(test_dataset)
model.save('Q2_2_c_Model')
print(score)