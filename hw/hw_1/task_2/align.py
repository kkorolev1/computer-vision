from scipy.signal import correlate2d
from skimage.io import imread, imshow
import numpy as np


def crop_image(img, crop_rate):
    cut_h = int(img.shape[0] * crop_rate)
    cut_w = int(img.shape[1] * crop_rate)
    return img[cut_h: img.shape[0] - cut_h, cut_w: img.shape[1] - cut_w]


def mse(img_pred, img_gt):
    return ((img_pred - img_gt)**2).sum() / (img_pred.shape[0] * img_pred.shape[1])


def intersect_images(img1, img2, x, y):
    left_cut, right_cut, top_cut, bottom_cut = (max(-y, 0), max(y, 0), max(-x, 0), max(x, 0))
    height, width = img1.shape
    out_img1 = img1[top_cut:height-bottom_cut, left_cut:width-right_cut]
    out_img2 = img2[bottom_cut:height-top_cut, right_cut:width-left_cut]
    return out_img1, out_img2


def mse_find_shift(img1, img2, shift_x_range, shift_y_range):
    """
    :param img1: channel to shift
    :param img2: base channel
    :return: row_shift, col_shift
    """

    min_shift = (0, 0)
    min_err = float('inf')

    for x in np.arange(*shift_x_range):
        for y in np.arange(*shift_y_range):
            img1_shifted, img2_cropped = intersect_images(img1, img2, x, y)

            mse_value = mse(img1_shifted, img2_cropped)
            if mse_value < min_err:
                min_err = mse_value
                min_shift = (x, y)

    return min_shift


def cross_cor_find_shift(img1, img2, shift_x, shift_y):
    """
    :param img1: channel to shift
    :param img2: base channel
    :return: row_shift, col_shift
    """

    base_channel = np.zeros((img2.shape[0] + 2*shift_x, img2.shape[1] + 2*shift_y))
    base_channel[shift_x:img2.shape[0]+shift_x, shift_y:img2.shape[1]+shift_y] = img2

    correlated_img = correlate2d(base_channel, img1, mode='valid')

    row_idx, col_idx = np.where(correlated_img == np.amax(correlated_img))
    row_shift, col_shift = row_idx[0] - shift_x, col_idx[0] - shift_y

    return row_shift, col_shift


def find_shift(img1, img2, shift_x_range, shift_y_range):
    """
    :param img1: channel to shift
    :param img2: base channel
    :return: row_shift, col_shift
    """
    #    return cross_cor_find_shift(img1, img2, shift_x, shift_y)
    return mse_find_shift(img1, img2, shift_x_range, shift_y_range)


def pyramid(img1, img2, threshold=500):
    """
    :param img1: channel to shift
    :param img2: base channel
    :return: row_shift, col_shift
    """
    if max(img1.shape) < threshold:
        return find_shift(img1, img2, (-15, 16), (-15, 16))

    cropped_img1 = img1[::2, ::2]
    cropped_img2 = img2[::2, ::2]
    shift_x_range, shift_y_range = pyramid(cropped_img1, cropped_img2, threshold)
    return find_shift(img1, img2, (2*shift_x_range-1, 2*shift_x_range+2), (2*shift_y_range-1, 2*shift_y_range+2))


def align(img, g_coord):
    img = img.astype('float64')
    channel_h = img.shape[0] // 3

    channel_b = img[:channel_h, :]
    channel_g = img[channel_h:2 * channel_h, :]
    channel_r = img[2 * channel_h:3 * channel_h, :]

    crop_rate = 0.07

    b = crop_image(channel_b, crop_rate)
    g = crop_image(channel_g, crop_rate)
    r = crop_image(channel_r, crop_rate)

    shift_b = np.array(pyramid(b, g))
    channel_b = np.roll(channel_b, shift_b, (0, 1))
    b_coord = np.array(g_coord) - shift_b - np.array((channel_h, 0))

    shift_r = np.array(pyramid(r, g))
    channel_r = np.roll(channel_r, shift_r, (0, 1))
    r_coord = np.array(g_coord) - shift_r + np.array((channel_h, 0))

    res_image = np.dstack((
        channel_r,
        channel_g,
        channel_b
    ))

    return res_image.astype('uint8'), b_coord, r_coord


if __name__ == '__main__':
    from skimage.io import imsave
    img = imread('tests/19_test_img_input/img.png')
    parts = open('tests/19_test_img_input/g_coord.csv').read().rstrip('\n').split(',')
    g_coord = (int(parts[0]), int(parts[1]))

    aligned_img, shift_b, shift_r = align(img, g_coord)

    imsave('aligned_img.png', aligned_img)

#%%

#%%
