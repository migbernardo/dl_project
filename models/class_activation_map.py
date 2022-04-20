from keras import backend as K
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow
from keras.models import load_model

if __name__ == '__main__':
    # load model (input the desired model's name)
    model = load_model(os.path.abspath('3_shallow_cnn_gmp.h5'))

    # change directory to raw images
    os.chdir(os.path.abspath(os.pardir))
    os.chdir(os.path.join(os.path.abspath(os.curdir), 'data', 'raw'))
    # save path to desired image (input class and image name)
    path_img = os.path.join(os.path.abspath(os.curdir), 'River', 'River_119.jpg')
    # open image
    img = Image.open(path_img)
    # convert image into rescaled tensor
    img = 1 / 255 * np.expand_dims(np.array(img), axis=0)
    # save prediction
    pred = model.predict(img)

    # get gradient
    with tensorflow.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_3')
        iterate = tensorflow.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(img)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # generate and plot class activation map
    heatmap = tensorflow.reduce_mean(tensorflow.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((25, 25))

    img2 = cv2.imread(path_img)

    heatmap = cv2.resize(heatmap, (img2.shape[1], img2.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    plt.imshow(heatmap)
    plt.imshow(cv2.imread(path_img), alpha=0.8)
    plt.show()
