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


def interpolate_channel(channel, mask):
    channel = channel.astype('float64')
    filter = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    NONE_VALUE = 1000
    channel[~mask] = NONE_VALUE

    # считаем количество непустых соседей для каждого пикселя
    num_neighbours = correlate2d(channel != NONE_VALUE, filter)[1: channel.shape[0]+1, 1: channel.shape[1]+1]
    num_neighbours[num_neighbours == 0] = 1
    channel[~mask] = 0

    # интерполяция для каждого пикселя
    interpolated_vals = correlate2d(channel, filter)[1: channel.shape[0]+1, 1: channel.shape[1]+1] // num_neighbours
    channel[~mask] = interpolated_vals[~mask]

    return channel


def bilinear_interpolation(colored_img):
    masks = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
    return np.dstack((
        interpolate_channel(colored_img[..., 0], masks[..., 0]),
        interpolate_channel(colored_img[..., 1], masks[..., 1]),
        interpolate_channel(colored_img[..., 2], masks[..., 2])
    )).astype('uint8')


def convolve(raw_img, i, j, filter, norm):
    return (raw_img[i - 2: i + 3, j - 2: j + 3] * filter).sum() / norm


def improved_interpolation(raw_img):
    channels = get_colored_img(raw_img).astype('float64')

    r = channels[..., 0]
    g = channels[..., 1]
    b = channels[..., 2]

    filter_gr = np.array([[0,0,-1,0,0], [0,0,2,0,0], [-1,2,4,2,-1], [0,0,2,0,0], [0,0,-1,0,0]])
    filter_gr_sum = filter_gr.sum()

    filter_gb = np.array([[0,0,-1,0,0], [0,0,2,0,0], [-1,2,4,2,-1], [0,0,2,0,0], [0,0,-1,0,0]])
    filter_gb_sum = filter_gb.sum()

    filter_rg_rb = np.array([[0,0,.5,0,0], [0,-1,0,-1,0], [-1,4,5,4,-1], [0,-1,0,-1,0], [0,0,.5,0,0]])
    filter_rg_rb_sum = filter_rg_rb.sum()

    filter_rg_br = np.array([[0,0,-1,0,0], [0,-1,4,-1,0], [.5,0,5,0,.5], [0,-1,4,-1,0], [0,0,-1,0,0]])
    filter_rg_br_sum = filter_rg_br.sum()

    filter_rb_bb = np.array([[0,0,-1.5,0,0], [0,2,0,2,0], [-1.5,0,6,0,-1.5], [0,2,0,2,0], [0,0,-1.5,0,0]])
    filter_rb_bb_sum = filter_rb_bb.sum()

    filter_bg_br = np.array([[0,0,.5,0,0], [0,-1,0,-1,0], [-1,4,5,4,-1], [0,-1,0,-1,0], [0,0,.5,0,0]])
    filter_bg_br_sum = filter_bg_br.sum()

    filter_bg_rb = np.array([[0,0,-1,0,0], [0,-1,4,-1,0], [.5,0,5,0,.5], [0,-1,4,-1,0], [0,0,-1,0,0]])
    filter_bg_rb_sum = filter_bg_rb.sum()

    filter_br_rr = np.array([[0,0,-1.5,0,0], [0,2,0,2,0], [-1.5,0,6,0,-1.5], [0,2,0,2,0], [0,0,-1.5,0,0]])
    filter_br_rr_sum = filter_br_rr.sum()

    for i in range(2, raw_img.shape[0] - 2):
        for j in range(2, raw_img.shape[1] - 2):
            if i % 2 == 0 and j % 2 == 0:
                r[i, j] = convolve(raw_img, i, j, filter_rg_rb, filter_rg_rb_sum)
                b[i, j] = convolve(raw_img, i, j, filter_bg_rb, filter_bg_rb_sum)
            elif i % 2 == 1 and j % 2 == 1:
                r[i, j] = convolve(raw_img, i, j, filter_rg_br, filter_rg_br_sum)
                b[i, j] = convolve(raw_img, i, j, filter_bg_br, filter_bg_br_sum)
            elif i % 2 == 0 and j % 2 == 1:
                g[i, j] = convolve(raw_img, i, j, filter_gr, filter_gr_sum)
                b[i, j] = convolve(raw_img, i, j, filter_br_rr, filter_br_rr_sum)
            else:
                g[i, j] = convolve(raw_img, i, j, filter_gb, filter_gb_sum)
                r[i, j] = convolve(raw_img, i, j, filter_rb_bb, filter_rb_bb_sum)

    '''for i in range(2, raw_img.shape[0] - 2, 2):
        for j in range(3, raw_img.shape[1] - 2, 2):
            g[i, j] = convolve(raw_img, i, j, filter_gr, filter_gr_sum)
            b[i, j] = convolve(raw_img, i, j, filter_br_rr, filter_br_rr_sum)

    for i in range(3, raw_img.shape[0] - 2, 2):
        for j in range(2, raw_img.shape[1] - 2, 2):
            g[i, j] = convolve(raw_img, i, j, filter_gb, filter_gb_sum)
            r[i, j] = convolve(raw_img, i, j, filter_rb_bb, filter_rb_bb_sum)

    for i in range(2, raw_img.shape[0] - 2, 2):
        for j in range(2, raw_img.shape[1] - 2, 2):
            r[i, j] = convolve(raw_img, i, j, filter_rg_rb, filter_rg_rb_sum)
            b[i, j] = convolve(raw_img, i, j, filter_bg_rb, filter_bg_rb_sum)
  
    for i in range(3, raw_img.shape[0] - 2, 2):
        for j in range(3, raw_img.shape[1] - 2, 2):
            r[i, j] = convolve(raw_img, i, j, filter_rg_br, filter_rg_br_sum)
            b[i, j] = convolve(raw_img, i, j, filter_bg_br, filter_bg_br_sum)'''

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    return np.dstack((r, g, b)).astype('uint8')


def mse(img_pred, img_gt):
    return np.linalg.norm(img_pred - img_gt)**2 / (img_pred.shape[0] * img_pred.shape[1] * img_pred.shape[2])


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype('float64')
    img_gt = img_gt.astype('float64')
    mse_err = mse(img_pred, img_gt)
    if mse_err == 0:
       raise ValueError("compute_psnr: MSE is 0")
    return 10 * np.log10((np.max(img_gt) ** 2) / mse_err)


if __name__ == "__main__":
    raw_img = np.array([[8, 5, 3, 7, 1, 3],
                     [5, 2, 6, 8, 8, 1],
                     [9, 9, 8, 1, 6, 4],
                     [9, 4, 2, 3, 6, 8],
                     [5, 4, 3, 2, 8, 7],
                     [7, 3, 3, 6, 9, 3]], dtype='uint8')
    improved_interpolation(raw_img)