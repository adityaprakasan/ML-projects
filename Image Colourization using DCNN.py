import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb

# Data Preprocessing
def generate_data(batch_size):
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)
    for batch in datagen.flow_from_directory('data/', batch_size=batch_size, target_size=(256, 256), class_mode=None):
        lab_batch = np.array([rgb2lab(img) for img in batch])
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128  # Normalize the values to [-1, 1]
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Model Building
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(generate_data(batch_size=20), epochs=10, steps_per_epoch=100)

# To Test on your own images
test_img = img_to_array(load_img('path_to_your_image.jpg'))
test_img = np.array(test_img, dtype=float)
gray_img = rgb2lab(1.0/255*test_img)[:,:,0]
gray_img = gray_img.reshape(1, 256, 256, 1)
output = model.predict(gray_img)
output = output * 128
result = np.zeros((256, 256, 3))
result[:,:,0] = gray_img[0][:,:,0]
result[:,:,1:] = output[0]
img = lab2rgb(result)
plt.imshow(img)
plt.show()
