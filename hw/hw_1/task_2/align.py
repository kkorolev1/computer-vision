from skimage.io import imread, imsave, imshow
import numpy as np
#import matplotlib.pyplot as plt


def crop_image(img, crop_rate):
    return img[int(img.shape[0] * crop_rate): int(img.shape[0] * (1 - crop_rate)) + 1,
           int(img.shape[1] * crop_rate): int(img.shape[1] * (1 - crop_rate)) + 1]


def mse(img1, img2):
    return np.linalg.norm(img1 - img2, ord='norm') / img1.shape[0] / img1.shape[1]


def mse_shift_images(img1, img2, shift_range=15):
    shift_values = np.arange(-shift_range, shift_range + 1)

    min_shift = 0
    min_err = float('inf')

    for shift_x in shift_values:
        for shift_y in shift_values:
            img2_shifted = img2[shift_x:img2.shape[0] + shift_x, shift_y:img2.shape[1] + shift_y]

            mse_value = mse(img1, img2_shifted)
            if mse_value < min_err:
                min_err = mse_value
                min_shift = (shift_x, shift_y)

    return min_shift


def align(img, coord):
    channel_h = img.shape[0] // 3

    channel_1 = img[:channel_h, :]
    channel_2 = img[channel_h:2 * channel_h, :]
    channel_3 = img[2 * channel_h:3 * channel_h, :]

    crop_rate = 0.07

    # channel_1 = crop_image(channel_1, crop_rate)
    # channel_2 = crop_image(channel_2, crop_rate)
    # channel_3 = crop_image(channel_3, crop_rate)

    #mse_shift_images(channel_1, channel_2)

    imshow(np.dstack((channel_3, channel_2, channel_1)))


img = imread('tests/00_test_img_input/img.png')
parts = open('tests/00_test_img_input/g_coord.csv').read().rstrip('\n').split(',')
g_coord = (int(parts[0]), int(parts[1]))

align(img, g_coord)

#%%
