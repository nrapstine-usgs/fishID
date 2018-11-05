import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize

from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model

def plot_inputs(files_img, nplot):
    ''' Preview image inputs
    Parameters
    ----------
    files_img : list
        image file names as strings in a list
    nplot : int
        number of plots in preview
    '''
    nx = len(files_img)
    temp_idx = np.random.randint(0, nx, nplot)

    plt.figure(figsize = (nplot, 3))
    for i, j in enumerate(temp_idx):
        img_temp = imread(files_img[j])

        plt.subplot(2, nplot, i+1)
        plt.imshow(img_temp)

    plt.show()

def get_unet(img_height, img_width, img_channels, activation='relu'):
    ''' Construct UNet CNN for segmenting in an image
    Parameters
    ----------
    img_height : int
        image height in pixels
    img_width : int
        image width in pixels
    img_channels : int
        number of image channels (e.g. RGB has 3 image channels)
    activation : string
        activation function as string for all layers
    Returns
    -------
    unet : keras.models.Model()
        The UNet CNN model described in Ronneberger et al. (2015)
    '''

    inputs = Input((img_height, img_width, img_channels))

    c1 = Conv2D(8, (3, 3), activation=activation, padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation=activation, padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation=activation, padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation=activation, padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation=activation, padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation=activation, padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation=activation, padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation=activation, padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation=activation, padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation=activation, padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation=activation, padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation=activation, padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation=activation, padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation=activation, padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation=activation, padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation=activation, padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation=activation, padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation=activation, padding='same') (c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    unet = Model(inputs=[inputs], outputs=[outputs])

    return unet
