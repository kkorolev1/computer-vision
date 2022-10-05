import numpy as np
from scipy.signal import convolve2d


def delete_vert_seam(image, mask):
    return image[~mask.astype('bool')].reshape(image.shape[0], image.shape[1] - 1)


def delete_seam(image, mask, axis):
    if len(image.shape) == 3:
        return np.dstack((delete_seam(image[..., 0], mask, axis),
                          delete_seam(image[..., 1], mask, axis),
                          delete_seam(image[..., 2], mask, axis)))
    if axis == 0:
        return delete_vert_seam(image.T, mask.T).T
    return delete_vert_seam(image, mask)


def get_energy(input_image):
    x_filter = np.array([[0,0,0], [-1,0,1], [0,0,0]])
    y_filter = np.array([[0,-1,0], [0,0,0], [0,1,0]])

    dx = convolve2d(input_image, x_filter[::-1, ::-1], boundary="symm", mode="same")
    dy = convolve2d(input_image, y_filter[::-1, ::-1], boundary="symm", mode="same")

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


def shrink(input_image, mask, axis):
    seam_mask = find_seam(input_image, mask, axis)
    return delete_seam(input_image, seam_mask, axis), delete_seam(mask, seam_mask, axis), seam_mask


def add_vert_seam(image, mask, constant=0):
    new_image = np.pad(image, ((0,0),(0,1)))

    new_image[:, -1] = new_image[:, -2]
    new_mask = np.pad(mask, ((0,0), (0,1)))

    new_seam_mask = np.roll(new_mask, 1, axis=1)
    new_seam_values = (new_image * (new_mask + new_seam_mask)).sum(axis=1) / 2
    seam_image = new_seam_mask * new_seam_values.reshape(-1,1)

    cumsum = np.cumsum(mask[::-1,::-1], axis=1)[::-1,::-1]

    left_image = np.pad(image * cumsum, ((0,0),(0,1)))

    new_image[:, -1] = 0
    right_image = np.roll(new_image * np.pad(1 - cumsum, ((0,0),(0,1))), 1, axis=1)

    return (left_image + seam_image + right_image) + new_mask * constant


def add_seam(image, mask, axis, constant=0):
    if len(image.shape) == 3:
        return np.dstack((add_seam(image[..., 0], mask, axis, constant),
                          add_seam(image[..., 1], mask, axis, constant),
                          add_seam(image[..., 2], mask, axis, constant)))
    if axis == 0:
        return add_vert_seam(image.T, mask.T, constant).T
    return add_vert_seam(image, mask, constant)


def expand(input_image, mask, axis):
    seam_mask = find_seam(input_image, mask, axis)
    return add_seam(input_image, seam_mask, axis), \
           add_seam(mask, seam_mask, axis), \
           seam_mask


def seam_carve(input_image, mode, mask=None):
    """
    :param input_image: Input image
    :param mode: Algorithm mode: horizontal shrink, vertical shrink, horizontal expand, vertical expand
    :param mask: Mask for image. -1 means deletion, 1 means conservation, 0 means no energy is changed
    :return:
    """

    if mask is None:
        mask = np.zeros(input_image.shape[:2])

    axis = 1 if "horizontal" in mode else 0

    if "shrink" in mode:
        return shrink(input_image, mask, axis)
    else:
        return expand(input_image, mask, axis)


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
    imshow(img)
    mask = convert_img_to_mask(imread("tests/01_test_img_input/mask.png"))
    new_image, new_mask, seam_mask = seam_carve(img, 'horizontal shrink', mask)

#%%
