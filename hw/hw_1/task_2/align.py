import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import numpy as np


def crop_image(img, crop_rate):
    return img[int(img.shape[0] * crop_rate): int(img.shape[0] * (1 - crop_rate)) + 1,
           int(img.shape[1] * crop_rate): int(img.shape[1] * (1 - crop_rate)) + 1]


def mse(img_pred, img_gt):
    return np.linalg.norm(img_pred - img_gt)**2 / (img_pred.shape[0] * img_pred.shape[1])


def mse_shift_images(img1, img2, shift_range=15):
    base = 2 * shift_range
    shift_values = np.arange(-shift_range, shift_range + 1)

    min_shift = 0
    min_err = float('inf')
    res_img = None

    for shift_x in shift_values:
        for shift_y in shift_values:
            img2_shifted = img2[base+shift_x:base+img2.shape[0] + shift_x, base+shift_y:base+img2.shape[1] + shift_y]
            img1_cropped = img1[:img2_shifted.shape[0], :img2_shifted.shape[1]]

            mse_value = mse(img1_cropped, img2_shifted)
            if mse_value < min_err:
                min_err = mse_value
                min_shift = (shift_x, shift_y)
                res_img = img2_shifted

    return min_err, np.array(min_shift), res_img


def align(img, g_coord):
    channel_h = img.shape[0] // 3

    channel_1 = img[:channel_h, :]
    channel_2 = img[channel_h:2 * channel_h, :]
    channel_3 = img[2 * channel_h:3 * channel_h, :]

    crop_rate = 0.07

    channel_1 = crop_image(channel_1, crop_rate)
    channel_2 = crop_image(channel_2, crop_rate)
    channel_3 = crop_image(channel_3, crop_rate)

    min_err_gr, min_shift_gr, channel_r = mse_shift_images(channel_2, channel_1)
    min_err_gb, min_shift_gb, channel_b = mse_shift_images(channel_2, channel_3)

    img_size = np.min([channel_r.shape, channel_b.shape], axis=1)

    res_image = np.dstack((
        channel_r[:img_size[0], :img_size[1]],
        channel_2[:img_size[0], :img_size[1]],
        channel_b[:img_size[0], :img_size[1]]
    ))

    return res_image.astype('uint8'), np.array(g_coord) + min_shift_gb, np.array(g_coord) + min_shift_gr


img = imread('tests/03_test_img_input/img.png')
parts = open('tests/03_test_img_input/g_coord.csv').read().rstrip('\n').split(',')
g_coord = (int(parts[0]), int(parts[1]))

align(img, g_coord)

#%%
