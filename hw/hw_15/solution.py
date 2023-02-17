from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """

            return parameter - parameter_grad * self.lr

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        return np.maximum(0, inputs)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        out = np.zeros(self.forward_inputs.shape, dtype=np.float)
        out[self.forward_inputs >= 0] = 1
        return grad_outputs * out


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        maximums = np.max(inputs, axis=1).reshape(-1, 1)
        return np.exp(inputs - maximums) / np.sum(np.exp(inputs - maximums), axis=1).reshape(-1, 1)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        softmax = self.forward_impl(self.forward_inputs)
        return grad_outputs * softmax - softmax * (grad_outputs * softmax).sum(axis=1).reshape(-1, 1)


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        return inputs @ self.weights + self.biases

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        self.weights_grad += self.forward_inputs.T @ grad_outputs
        self.biases_grad += grad_outputs.sum(axis=0)
        return grad_outputs @ self.weights.T


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        y_pred = np.maximum(eps, y_pred)
        return -np.array([(y_gt * np.log(y_pred)).sum(axis=1).mean()])

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        y_pred = np.maximum(eps, y_pred)
        return -y_gt / y_pred / y_gt.shape[0]


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    optimizer = SGD(lr=1e-3)
    model = Model(loss, optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(input_shape=(784,), units=784))
    model.add(ReLU())
    model.add(Dense(units=784))
    model.add(ReLU())
    model.add(Dense(units=784))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=256, epochs=10, shuffle=False, verbose=True, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # делаем кросс-корреляцию с обратным ядром = свертка
    kernels = kernels[..., ::-1, ::-1]

    output_shape = (inputs.shape[2] + 2 * padding - kernels.shape[2] + 1, inputs.shape[3] + 2 * padding - kernels.shape[3] + 1)
    output = np.zeros((inputs.shape[0], kernels.shape[0], *output_shape))

    padded_input = np.zeros((*inputs.shape[:2], inputs.shape[2] + 2 * padding, inputs.shape[3] + 2 * padding))
    padded_input[..., padding: padding + inputs.shape[2], padding: padding + inputs.shape[3]] = inputs

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output[..., i, j] = (padded_input[..., i: i + kernels.shape[2], j: j + kernels.shape[3]][:, None, :, :, :] * kernels).sum(axis=(-1, -2, -3)).reshape((inputs.shape[0], kernels.shape[0]))

    return output


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        return convolve(inputs, self.kernels, (self.kernel_size - 1) // 2) + self.biases[None, :, None, None]

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        #self.kernels_grad = convolve(grad_outputs, )
        #self.biases_grad = grad_outputs.sum(axis=[2, 3])

        return ...


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        return ...
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        return ...
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # Training
        if self.is_training:
            mean = inputs.mean(axis=(0, 2, 3), keepdims=True)
            self.forward_centered_inputs = inputs - mean
            var = inputs.var(axis=(0, 2, 3), keepdims=True)
            self.forward_inverse_std = 1 / np.sqrt(var.ravel() + eps)
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std[:, None, None]

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.ravel()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.ravel()
        else:
            # Evaluation
            self.forward_normalized_inputs = (inputs - self.running_mean[:, None, None]) / np.sqrt(self.running_var[:, None, None] + eps)

        return self.gamma[:, None, None] * self.forward_normalized_inputs + self.beta[:, None, None]

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        n, d, h, w = grad_outputs.shape
        size = n * h * w
        self.gamma_grad = (grad_outputs * self.forward_normalized_inputs).sum(axis=(0, 2, 3))
        self.beta_grad = grad_outputs.sum(axis=(0, 2, 3))
        df_xhat = grad_outputs * self.gamma[None, :, None, None]

        return self.forward_inverse_std[None, :, None, None] / size * (size * df_xhat - self.forward_normalized_inputs * (df_xhat * self.forward_normalized_inputs).sum(
            axis=(0, 2, 3), keepdims=True) - df_xhat.sum(axis=(0, 2, 3), keepdims=True))


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        return inputs.reshape(-1, self.output_shape[0])

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        return grad_outputs.reshape(grad_outputs.shape[0], *self.input_shape)


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        if self.is_training:
            self.forward_mask = np.random.uniform(0, 1, size=inputs.shape) >= self.p
            return self.forward_mask * inputs
        else:
            return inputs * (1 - self.p)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        return grad_outputs * self.forward_mask


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = ...

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(...)
    model.add(...)
    ...
    model.add(...)
    model.add(...)

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(...)

    # your code here /\
    return model

# ============================================================================