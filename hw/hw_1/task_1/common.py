import numpy as np

def assert_value_is_ndarray(value):
    __tracebackhide__ = True
    assert isinstance(value, (np.ndarray, np.generic)), (
        f"Value should be an instance of np.ndarray, but it is {type(value)}."
    )


def assert_dtypes_compatible(actual_dtype, correct_dtype):
    __tracebackhide__ = True
    assert (
            np.can_cast(actual_dtype, correct_dtype, casting='same_kind') and
            np.can_cast(correct_dtype, actual_dtype, casting='same_kind')
    ), (
        "The dtypes of actual value and correct value are not the same "
        "and can't be safely converted.\n"
        f"actual.dtype={actual_dtype}, correct.dtype={correct_dtype}"
    )


def assert_shapes_match(actual_shape, correct_shape):
    __tracebackhide__ = True
    assert (
            len(actual_shape) == len(correct_shape) and
            actual_shape == correct_shape
    ), (
        "The shapes of actual value and correct value are not the same.\n"
        f"actual.shape={actual_shape}, correct.shape={correct_shape}"
    )


def assert_ndarray_equal(*, actual, correct, rtol=0, atol=1e-6, err_msg=''):
    __tracebackhide__ = True
    assert_value_is_ndarray(actual)
    assert_dtypes_compatible(actual.dtype, correct.dtype)
    assert_shapes_match(actual.shape, correct.shape)
    np.testing.assert_allclose(actual, correct, atol=atol, rtol=rtol,
                               verbose=True, err_msg=err_msg)
