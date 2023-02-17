import numpy as np
import scipy
import scipy.fft


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    dx = np.arange(-(size - 1) // 2, (size - 1) // 2 + 1)
    mat = np.exp(-dx*dx/(2*sigma*sigma)).reshape(-1, 1)
    kernel = (mat @ mat.T) / (2*np.pi*sigma*sigma)
    return kernel / kernel.sum()


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    h = np.array(h)

    pad_h = shape[0] - h.shape[0]
    pad_w = shape[1] - h.shape[1]

    pad_h_half = 0, pad_h
    pad_w_half = 0, pad_w

    h_ext = np.pad(h, (pad_h_half, pad_w_half))
    return scipy.fft.fft2(h_ext)


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros_like(H)

    mask = np.abs(H) > threshold
    H_inv[mask] = 1 / H[mask]

    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H_inv = inverse_kernel(fourier_transform(h, blurred_img.shape), threshold)
    return np.abs(np.fft.ifft2(G * H_inv))


def wiener_filtering(blurred_img, h, K=0.001):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    F_est = G * H.conjugate() / (np.abs(H)**2 + K)
    return np.abs(np.fft.ifft2(F_est))


def mse(img1, img2):
    return np.sum((img1 - img2)**2) / (img1.shape[0] * img1.shape[1])


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    mse_err = mse(img1.astype('float64'), img2.astype('float64'))
    return 20 * np.log10(255 / np.sqrt(mse_err))