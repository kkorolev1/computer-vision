import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2gray


def shrink(input_image, is_horizontal):
    width = input_image.shape[1]
    height = input_image.shape[0]

    x_filter = np.array([[0,0,0], [-1,0,1], [0,0,0]])
    y_filter = np.array([[0,-1,0], [0,0,0], [0,1,0]])

    dx = convolve2d(input_image, x_filter, boundary="symm")
    dy = convolve2d(input_image, y_filter, boundary="symm")

    energy = np.sqrt(dx ** 2 + dy**2)

    carve_min = np.zeros_like(input_image)

    if is_horizontal:
        carve_min[0, :] = input_image[0, :]
        min_path = []
        for i in range(1, input_image.shape[0]):
            for j in range(input_image.shape[1]):
                neigh = [energy[i - 1, j - 1] if j > 0 else float("inf"), energy[i - 1, j], energy[i - 1, j + 1] if j > 0 else float("inf")]
                idx = np.argmax(neigh)
                min_path.append(idx)
                carve_min[i, j] = neigh[idx] + energy[i, j]

        print(min_path)

def seam_carve(input_image, mode, mask=None):
    """
    :param input_image: Input image
    :param mode: Algorith mode: horizontal shrink, vertical shrink, horizontal expand, vertical expand
    :param mask: Mask for image. -1 means deletion, 1 means conserve, 0 means no energy is changed
    :return:
    """
    input_image = rgb2gray(input_image).astype('float32')

    if mode == "horizontal shrink" or mode == "vertical shrink":
        shrink(input_image, mode == "horizontal shrink")



if __name__ == "__main__":
    from skimage.io import imread, imshow
    img = imread("tests/01_test_img_input/img.png")
    res = seam_carve(img, 'horizontal shrink')
    #imshow(res, cmap='gray')