import numpy as np
from scipy.signal import correlate2d
from skimage import img_as_ubyte


def red_channel_mask(n_rows, n_cols):
    base_mask = np.dstack((np.tile([0, 1], n_cols // 2 + (n_cols % 2 == 1))[:n_cols], np.zeros(n_cols)))
    return np.tile(base_mask, n_rows // 2 + (n_rows % 2 == 1))[0].T[:n_rows, :]


def green_channel_mask(n_rows, n_cols):
    base_mask = np.dstack((np.tile([1, 0], n_cols // 2 + (n_cols % 2 == 1))[:n_cols], np.tile([0, 1], n_cols // 2 + (n_cols % 2 == 1))[:n_cols]))
    return np.tile(base_mask, n_rows // 2 + (n_rows % 2 == 1))[0].T[:n_rows, :]


def blue_channel_mask(n_rows, n_cols):
    base_mask = np.dstack((np.zeros(n_cols), np.tile([1, 0], n_cols // 2 + (n_cols % 2 == 1))[:n_cols]))
    return np.tile(base_mask, n_rows // 2 + (n_rows % 2 == 1))[0].T[:n_rows, :]


def get_bayer_masks(n_rows, n_cols):
    return np.dstack((red_channel_mask(n_rows, n_cols),
                      green_channel_mask(n_rows, n_cols),
                      blue_channel_mask(n_rows, n_cols))).astype('bool')


def get_colored_img(raw_img):
    masks = get_bayer_masks(*raw_img.shape)
    return np.dstack([raw_img * masks[..., channel_idx] for channel_idx in range(3)])


def interpolate_channel(channel):
    filter = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # считаем количество ненулевых соседей для каждого пикселя
    num_nonzeros = correlate2d(channel > 0, filter)[1:1+channel.shape[0], 1:1+channel.shape[1]]
    num_nonzeros[num_nonzeros == 0] = 1

    # интерполяция для каждого пикселя
    interpolated_vals = correlate2d(channel, filter)[1:1+channel.shape[0], 1:1+channel.shape[1]] / num_nonzeros

    np.putmask(channel, channel == 0, np.around(interpolated_vals).astype('uint8'))
    return channel


def bilinear_interpolation(colored_img):
    return np.dstack((
        interpolate_channel(colored_img[..., 0]),
        interpolate_channel(colored_img[..., 1]),
        interpolate_channel(colored_img[..., 2])
    ))


def improved_interpolation(raw_img):
    # (канал, тип пикселя в шаблоне)
    filter_gr = np.array([[0,0,-1,0,0], [0,0,2,0,0], [-1,2,4,2,-1], [0,0,2,0,0], [0,0,-1,0,0]])
    filter_gb = np.array([[0,0,-1,0,0], [0,0,2,0,0], [-1,2,4,2,-1], [0,0,2,0,0], [0,0,-1,0,0]])

    filter_rg_rb = np.array([[0,0,.5,0,0], [0,-1,0,-1,0], [-1,4,5,4,-1], [0,-1,0,-1,0], [0,0,.5,0,0]])
    filter_rg_br = np.array([[0,0,.5,0,0], [0,-1,0,-1,0], [-1,4,5,4,-1], [0,-1,0,-1,0], [0,0,.5,0,0]])
    filter_rb_bb = np.array([[0,0,-1.5,0,0], [0,2,0,2,0], [-1.5,0,6,0,-1.5], [0,2,0,2,0], [0,0,-1.5,0,0]])

    filter_bg_br = np.array([[0,0,.5,0,0], [0,-1,0,-1,0], [-1,4,5,4,-1], [0,-1,0,-1,0], [0,0,.5,0,0]])
    filter_bg_rb = np.array([[0,0,.5,0,0], [0,-1,0,-1,0], [-1,4,5,4,-1], [0,-1,0,-1,0], [0,0,.5,0,0]])
    filter_br_rr = np.array([[0,0,-1.5,0,0], [0,2,0,2,0], [-1.5,0,6,0,-1.5], [0,2,0,2,0], [0,0,-1.5,0,0]])


def mse(img_pred, img_gt):
    return np.linalg.norm(img_pred - img_gt)**2 / (img_pred.shape[0] * img_pred.shape[1] * img_pred.shape[2])


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype('float64')
    img_gt = img_gt.astype('float64')
    mse_err = mse(img_pred, img_gt)
    if mse_err == 0:
       raise ValueError("compute_psnr: MSE is 0")
    return 10 * np.log10((np.max(img_gt) ** 2) / mse_err)


from skimage.io import imread, imshow
import matplotlib.pyplot as plt
#colored_img = get_colored_img(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='uint8'))
#print(colored_img[..., 0])
#print(colored_img[..., 1])
#print(colored_img[..., 2])
#interp = bilinear_interpolation(colored_img)
#print(interp[1, 1])

raw_img = img_as_ubyte(imread('tests/04_unittest_bilinear_img_input/02.png'))
img = img_as_ubyte(bilinear_interpolation(get_colored_img(raw_img)))
#imshow(img)

gt_img = img_as_ubyte(imread('tests/04_unittest_bilinear_img_input/gt_02.png'))

r = slice(1, -1), slice(1, -1)

#f, axes = plt.subplots(1, 2)
#axes[0].imshow(img[r])
#axes[1].imshow(gt_img[r])


img_pred = np.array([[[146, 222, 187],
                   [254, 123,  38],
                   [ 57, 255, 135]],
                  [[230, 176, 213],
                   [114,  38, 184],
                   [ 47, 212,  52]],
                  [[100, 111, 170],
                   [ 52, 230, 182],
                   [213,  50, 197]]], dtype='uint8')
img_gt = np.array([[[254,  60,   6],
                 [216,  53,  14],
                 [106, 185, 239]],
                [[121,  34,  29],
                 [ 49, 139, 149],
                 [  6, 159, 221]],
                [[240,  53, 124],
                 [  3, 194, 227],
                 [ 84,  12, 218]]], dtype='uint8')

print(compute_psnr(img_pred, img_gt))