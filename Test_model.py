import numpy as np
import os
import tensorflow as tf
from keras.models import load_model

model = load_model('model.h5')

result = []
batch_size = 32
img_height = 180
img_width = 180

test_dir = r"Test/"

class_name = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 'tulip']

for imageName in os.listdir(test_dir):
    image = tf.keras.utils.load_img(
        test_dir + imageName, target_size=(img_height, img_width)
    )
    image_arr = tf.keras.utils.img_to_array(image)
    image_arr = tf.expand_dims(image_arr, 0)
    predictions = model.predict(image_arr)
    score = tf.nn.softmax(predictions[0])
    result.append(class_name[np.argmax(score)])

print(result)


