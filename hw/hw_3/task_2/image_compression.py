import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы и проекция матрицы на новое пр-во
    """
    
    # Your code here
    
    # Отцентруем каждую строчку матрицы
    ...
    # Найдем матрицу ковариации
    ...
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    ...
    # Посчитаем количество найденных собственных векторов
    ...
    # Сортируем собственные значения в порядке убывания
    ...
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    ...
    # Оставляем только p собственных векторов
    ...
    # Проекция данных на новое пространство

    # matrix.sum(axis=1) / (matrix != 0).sum(axis=1)
    means = matrix.mean(axis=1)
    centered_matrix = matrix - means[:, None]

    cov_matrix = np.cov(centered_matrix)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    indices = np.argsort(-eigen_values)[:p]
    eigen_vectors = eigen_vectors[:, indices]

    return eigen_vectors, eigen_vectors.T @ centered_matrix, means


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        
        # Your code here
        result_img.append(np.clip(comp[0] @ comp[1] + comp[2][:, None], 0, 255).astype(np.uint8))

    return np.stack(result_img, axis=2)


def pca_visualize():
    plt.clf()
    img = imread('me.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            # Your code here
            compressed.append(pca_compression(img[..., j], p))

        axes[i // 3, i % 3].imshow(pca_decompression(compressed))
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    Cb = 128 - 0.1687 * img[..., 0] - 0.3313 * img[..., 1] + 0.5 * img[..., 2]
    Cr = 128 + 0.5 * img[..., 0] - 0.4187 * img[..., 1] - 0.0813 * img[..., 2]

    return np.dstack((Y, Cb, Cr))


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    Y = img[..., 0]
    Cb = img[..., 1] - 128
    Cr = img[..., 2] - 128

    R = Y + 1.402 * Cr
    G = Y - 0.34414 * Cb - 0.71414 * Cr
    B = Y + 1.77 * Cb
    
    return np.dstack((R, G, B))


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr = rgb2ycbcr(rgb_img)
    sigma = 10

    res_img = ycbcr2rgb(np.dstack(
        (ycbcr[..., 0],
         gaussian_filter(ycbcr[..., 1], sigma=sigma),
         gaussian_filter(ycbcr[..., 2], sigma=sigma))))

    res_img = np.clip(res_img, 0, 255).astype(np.uint8)
    plt.imshow(res_img)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    ycbcr = rgb2ycbcr(rgb_img)
    sigma = 10

    res_img = ycbcr2rgb(np.dstack((gaussian_filter(ycbcr[..., 0], sigma=sigma), ycbcr[..., 1], ycbcr[..., 2])))
    res_img = np.clip(res_img, 0, 255).astype(np.uint8)
    plt.imshow(res_img)

    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """

    return gaussian_filter(component, 10)[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    mat = block.astype(np.float32)

    u = np.arange(0, 8)
    cos_mat = np.cos(((2*u + 1)*np.pi/16).reshape(-1, 1) @ u.reshape(1, -1)) / 2

    alpha = np.ones((8, 8))
    alpha[:, 0] = 1/np.sqrt(2)

    return (alpha * cos_mat).T @ mat @ (alpha * cos_mat)

# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    
    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    s = 0

    if 1 <= q < 50:
        s = 5000 / q
    elif 50 <= q <= 99:
        s = 200 - 2*q
    else:
        s = 1

    quantization_matrix = default_quantization_matrix.astype(np.float32)
    res_quantization_matrix = (50 + s * quantization_matrix) / 100
    res_quantization_matrix = np.clip(res_quantization_matrix, 0, 255).astype(np.uint8)
    res_quantization_matrix[res_quantization_matrix == 0] = 1

    return res_quantization_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    res = []

    i, j = 0, 0
    res.append(block[i, j])

    while True:
        if j < 7:
            j += 1
        else:
            i += 1

        for k in range(8):
            res.append(block[i, j])
            if i < 7 and j > 0:
                i += 1
                j -= 1
            else:
                break

        if i < 7:
            i += 1
        else:
            j += 1

        for k in range(8):
            res.append(block[i, j])
            if i > 0 and j < 7:
                i -= 1
                j += 1
            else:
                break

        if i == 7 and j == 7:
            break

    return res


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    res_list = []
    num_zeros = 0

    for x in zigzag_list:
        if x != 0:
            if num_zeros > 0:
                res_list.append(num_zeros)
                num_zeros = 0
            res_list.append(x)
        else:
            if num_zeros == 0:
                res_list.append(x)
            num_zeros += 1

    if num_zeros > 0:
        res_list.append(num_zeros)

    return res_list


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here
    
    # Переходим из RGB в YCbCr
    ...
    # Уменьшаем цветовые компоненты
    ...
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    ...
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    ...
    ycbcr = rgb2ycbcr(img)

    y = ycbcr[..., 0]
    cb = downsampling(ycbcr[..., 1])
    cr = downsampling(ycbcr[..., 2])

    channels = [y, cb, cr]
    q_matrices = [quantization_matrixes[0], quantization_matrixes[1], quantization_matrixes[1]]

    compressed = []

    for channel_idx in range(3):
        compressed.append([])
        for i in range(0, channels[channel_idx].shape[0] - 7, 8):
            for j in range(0, channels[channel_idx].shape[1] - 7, 8):
                block = channels[channel_idx][i:i+8, j:j+8] - 128
                compressed[-1].append(compression(zigzag(quantization(dct(block), q_matrices[channel_idx]))))
    return compressed


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    res_list = []
    i = 0

    while i < len(compressed_list):
        res_list.append(compressed_list[i])
        if compressed_list[i] == 0:
            res_list += [0] * (compressed_list[i + 1] - 1)
            i += 1
        i += 1
    return res_list


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    res_mat = np.zeros((8, 8))

    i, j = 0, 0
    list_idx = 0

    res_mat[i, j] = input[list_idx]

    while True:
        if j < 7:
            j += 1
        else:
            i += 1

        list_idx += 1

        for k in range(8):
            res_mat[i, j] = input[list_idx]
            if i < 7 and j > 0:
                i += 1
                j -= 1
                list_idx += 1
            else:
                break

        if i < 7:
            i += 1
        else:
            j += 1
        list_idx += 1

        for k in range(8):
            res_mat[i, j] = input[list_idx]
            if i > 0 and j < 7:
                i -= 1
                j += 1
                list_idx += 1
            else:
                break

        if i == 7 and j == 7:
            break

    return res_mat


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    mat = block.astype(np.float32)

    u = np.arange(0, 8)
    cos_mat = np.cos(((2*u + 1)*np.pi/16).reshape(-1, 1) @ u.reshape(1, -1)) / 2

    alpha = np.ones((8, 8))
    alpha[:, 0] = 1/np.sqrt(2)

    res = (alpha * cos_mat) @ mat @ (alpha * cos_mat).T
    return np.round(res)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    res_mat = np.zeros((2 * component.shape[0], 2 * component.shape[1]))

    for i in range(0, component.shape[0]):
        res_mat[2*i, 0::2] = component[i, :]
        res_mat[2*i, 1::2] = component[i, :]
        res_mat[2*i+1, :] = res_mat[2*i, :]

    return res_mat


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    q_matrices = [quantization_matrixes[0], quantization_matrixes[1], quantization_matrixes[1]]
    sizes = [(result_shape[0], result_shape[1]), (result_shape[0] // 2, result_shape[1] // 2), (result_shape[0] // 2, result_shape[1] // 2)]

    channels = []

    for channel_idx in range(3):
        channel = np.zeros(sizes[channel_idx])
        for j in range(len(result[channel_idx])):
            result[channel_idx][j] = inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(result[channel_idx][j])), q_matrices[channel_idx]))

        block_idx = 0
        for i in range(0, sizes[channel_idx][0] - 7, 8):
            for j in range(0, sizes[channel_idx][1] - 7, 8):
                channel[i:i+8, j:j+8] = result[channel_idx][block_idx] + 128
                block_idx += 1

        channels.append(channel)

    channels[1] = upsampling(channels[1])
    channels[2] = upsampling(channels[2])

    return np.clip(ycbcr2rgb(np.dstack(channels)), 0, 255).astype(np.uint8)


def jpeg_visualize():
    plt.clf()
    img = imread('me.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    img = img[:(img.shape[0] // 8) * 8, :(img.shape[1] // 8) * 8, :]

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        y_qmat = own_quantization_matrix(y_quantization_matrix, p)
        cbcr_qmat = own_quantization_matrix(color_quantization_matrix, p)

        compressed = jpeg_compression(img, (y_qmat, cbcr_qmat))
        decompressed_img = jpeg_decompression(compressed, img.shape, (y_qmat, cbcr_qmat))

        axes[i // 3, i % 3].imshow(decompressed_img)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")

pca_visualize()