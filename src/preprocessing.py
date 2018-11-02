import numpy as np


def normalize_data(train_set,
                   mode="per channel", scale=None):
    """
    normalize images per channel, per pixel or with a fixed value
    Args:
        dataset: Should use train set only.
        return: Note that scale are also returns, which should be reused for
        test and validation set.
    """

    if scale is None:
        if mode == "per channel":
            n_channels = np.shape(train_set)[1]
            scale = np.std(train_set, axis=(0, 2, 3)).reshape(1, n_channels, 1, 1)
        elif mode == "per pixel":
            scale = np.std(train_set, 0)
        elif mode == "fixed value":
            scale = 255.
        else:
            raise ValueError("Specify mode of scaling (should be "
                             "'per channel', 'per pixel' or 'fixed value')")

    train_set /= scale
    return train_set


def global_contrast_normalization(dataset, scale="std"):
    """
    Subtract mean across features (pixels) and normalize by scale, which is
    either the standard deviation, l1- or l2-norm across features (pixel).
    That is, normalization for each sample (image) globally across features.
    """

    assert scale in ("std", "l1", "l2")

    na = np.newaxis

    dataset_mean = np.mean(dataset, axis=(1, 2, 3),
                           dtype=np.float32)[:, na, na, na]

    dataset -= dataset_mean

    if scale == "std":
        dataset_scale = np.std(dataset, axis=(1, 2, 3),
                               dtype=np.float32)[:, na, na, na]

    if scale == "l1":
        dataset_scale = np.sum(np.absolute(dataset), axis=(1, 2, 3),
                               dtype=np.float32)[:, na, na, na]

    if scale == "l2":
        # equivalent to "std" since mean is subtracted beforehand
        dataset_scale = np.sqrt(np.sum(dataset ** 2, axis=(1, 2, 3),
                                       dtype=np.float32))[:, na, na, na]

    dataset /= dataset_scale


def zca_whitening(X_train, X_val, X_test, eps=0.1):
    """
     Apply ZCA whitening. Epsilon parameter eps prevents division by zero.
    """

    # get shape to later reshape data to original format
    shape_train = X_train.shape
    shape_val = X_val.shape
    shape_test = X_test.shape

    if X_train.ndim > 2:
        X_train = X_train.reshape(shape_train[0], np.prod(shape_train[1:]))
        X_val = X_val.reshape(shape_val[0], np.prod(shape_val[1:]))
        X_test = X_test.reshape(shape_test[0], np.prod(shape_test[1:]))

    # center data
    means = np.mean(X_train, axis=0)
    X_train -= means
    X_val -= means
    X_test -= means

    # correlation matrix
    sigma = np.dot(X_train.T, X_train) / shape_train[0]

    # SVD
    U,S,V = np.linalg.svd(sigma)

    # ZCA Whitening matrix
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))

    # Whiten
    X_train = np.dot(X_train, ZCAMatrix.T)
    X_val = np.dot(X_val, ZCAMatrix.T)
    X_test = np.dot(X_test, ZCAMatrix.T)

    # reshape to original shape
    X_train = X_train.reshape(shape_train)
    X_val = X_val.reshape(shape_val)
    X_test = X_test.reshape(shape_test)

    return X_train, X_val, X_test