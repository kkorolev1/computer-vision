import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.io import imshow


def delete_vert_seam(image, idxs):
    mask = np.ones_like(image).astype('bool')
    mask[np.arange(image.shape[0]), idxs] = False
    return image[mask].reshape(image.shape[0], image.shape[1] - 1), ~mask


def delete_seam(image, idxs, axis):
    if axis == 0:
        new_image, mask = delete_vert_seam(image.T, idxs)
        return new_image.T, mask.T
    return delete_vert_seam(image, idxs)


def get_energy(input_image):
    x_filter = np.array([[0,0,0], [1,0,-1], [0,0,0]])
    y_filter = np.array([[0,1,0], [0,0,0], [0,-1,0]])

    dx = convolve2d(input_image, x_filter, boundary="symm", mode="same")
    dy = convolve2d(input_image, y_filter, boundary="symm", mode="same")

    return np.sqrt(dx ** 2 + dy**2).astype('float64')


def find_vert_seam(brightness, mask):
    energy = get_energy(brightness)
    energy += np.float64(energy.shape[0] * energy.shape[1] * 256.0) * mask

    seam_min = np.full(brightness.shape, np.inf).astype('float64')
    seam_min[0, :] = energy[0, :]

    prev = np.full(brightness.shape, -1)

    for i in range(1, brightness.shape[0]):
        for j in range(brightness.shape[1]):
            idx = seam_min[i - 1, max(0, j - 1): min(brightness.shape[1], j + 2)].argmin() - (1 if j > 0 else 0)
            prev[i, j] = j + idx
            seam_min[i, j] = seam_min[i - 1, j + idx] + energy[i, j]

    idx = np.argmin(seam_min[-1, :])
    seam_mask = np.zeros(brightness.shape, dtype='int8')

    for i in range(brightness.shape[0] - 1, -1, -1):
        if idx == -1:
            break
        seam_mask[i, idx] = 1
        idx = prev[i, idx]

    return seam_mask


def find_seam(input_image, mask, axis):
    brightness = (0.299 * input_image[:, :, 0] + 0.587 * input_image[:, :, 1] + 0.114 * input_image[:, :, 2]).astype("float64")
    return find_vert_seam(brightness, mask) if axis == 1 else find_vert_seam(brightness.T, mask.T).T


def seam_carve(input_image, mode, mask=None):
    """
    :param input_image: Input image
    :param mode: Algorithm mode: horizontal shrink, vertical shrink, horizontal expand, vertical expand
    :param mask: Mask for image. -1 means deletion, 1 means conservation, 0 means no energy is changed
    :return:
    """
    HORIZONTAL_SHRINK_MODE = "horizontal shrink"
    VERTICAL_SHRINK_MODE = "vertical shrink"
    HORIZONTAL_EXPAND_MODE = "horizontal expand"
    VERTICAL_EXPAND_MODE = "vertical expand"

    if mask is None:
        mask = np.zeros(input_image.shape[:2])

    if mode == HORIZONTAL_SHRINK_MODE or mode == VERTICAL_SHRINK_MODE:
        axis = 1 if mode == HORIZONTAL_SHRINK_MODE else 0
        seam_mask = find_seam(input_image, mask, axis=axis)
        return None, None, seam_mask
        #shrinked_mask, seam_mask = delete_seam(mask, seam, axis=axis)
        #return np.dstack((r, g, b)).astype('uint8'), shrinked_mask, seam_mask.astype('uint8')

    axis = 1 if mode == HORIZONTAL_EXPAND_MODE else 0
    seam_mask = find_seam(input_image, mask, axis=axis)
    return None, None, seam_mask


if __name__ == "__main__":
    def get_seam_coords(seam_mask):
        coords = np.where(seam_mask)
        t = [i for i in zip(coords[0], coords[1])]
        t.sort(key=lambda i: i[0])
        return tuple(t)

    def convert_img_to_mask(img):
        return ((img[:, :, 0] != 0) * -1 + (img[:, :, 1] != 0)).astype('int8')

    from skimage.io import imread, imshow
    from pickle import dump, load

    img = imread("tests/01_test_img_input/img.png")
    mask = convert_img_to_mask(imread("tests/01_test_img_input/mask.png"))
    new_image, new_mask, seam_mask = seam_carve(img, 'horizontal shrink', mask)

    with open("output.txt", "w") as f:
        np.savetxt(f, seam_mask, fmt='%d')

    with open("tests/01_test_img_gt/seams", "rb") as f:
        seams = load(f)
        mask = np.zeros(img.shape[:2])
        rows = [s[0] for s in seams]
        cols = [s[1] for s in seams]
        mask[rows, cols] = 1

        with open("gt.txt", "w") as f2:
            np.savetxt(f2, mask, fmt='%d')

    #print(seam_mask)
    #imshow(res[0])
#%%
