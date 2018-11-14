import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, PCA


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


def global_contrast_normalization(X_train, X_val, X_test, scale="std"):
    """
    Subtract mean across features (pixels) and normalize by scale, which is
    either the standard deviation, l1- or l2-norm across features (pixel).
    That is, normalization for each sample (image) globally across features.
    """

    assert scale in ("std", "l1", "l2")

    na = np.newaxis

    X_train_mean = np.mean(X_train, axis=(1, 2, 3),
                           dtype=np.float32)[:, na, na, na]
    X_val_mean = np.mean(X_val, axis=(1, 2, 3),
                         dtype=np.float32)[:, na, na, na]
    X_test_mean = np.mean(X_test, axis=(1, 2, 3),
                          dtype=np.float32)[:, na, na, na]

    X_train -= X_train_mean
    X_val -= X_val_mean
    X_test -= X_test_mean

    if scale == "std":
        X_train_scale = np.std(X_train, axis=(1, 2, 3),
                               dtype=np.float32)[:, na, na, na]
        X_val_scale = np.std(X_val, axis=(1, 2, 3),
                             dtype=np.float32)[:, na, na, na]
        X_test_scale = np.std(X_test, axis=(1, 2, 3),
                              dtype=np.float32)[:, na, na, na]
    if scale == "l1":
        X_train_scale = np.sum(np.absolute(X_train), axis=(1, 2, 3),
                               dtype=np.float32)[:, na, na, na]
        X_val_scale = np.sum(np.absolute(X_val), axis=(1, 2, 3),
                             dtype=np.float32)[:, na, na, na]
        X_test_scale = np.sum(np.absolute(X_test), axis=(1, 2, 3),
                              dtype=np.float32)[:, na, na, na]
    if scale == "l2":
        # equivalent to "std" since mean is subtracted beforehand
        X_train_scale = np.sqrt(np.sum(X_train ** 2, axis=(1, 2, 3),
                                       dtype=np.float32))[:, na, na, na]
        X_val_scale = np.sqrt(np.sum(X_val ** 2, axis=(1, 2, 3),
                                     dtype=np.float32))[:, na, na, na]
        X_test_scale = np.sqrt(np.sum(X_test ** 2, axis=(1, 2, 3),
                                      dtype=np.float32))[:, na, na, na]

    X_train /= X_train_scale
    X_val /= X_val_scale
    X_test /= X_test_scale
    return X_train, X_val, X_test


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


def pca(X_train, X_val, X_test, var_retained=0.95):
    """
    PCA such that var_retained of variance is retained (w.r.t. train set)
    """

    print("Applying PCA...")

    # reshape to 2D if input is tensor
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        if X_val.size > 0:
            X_val = X_val.reshape(X_val.shape[0], -1)
        if X_test.size > 0:
            X_test = X_test.reshape(X_test.shape[0], -1)

    pca = PCA(n_components=var_retained)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    if X_val.size > 0:
        X_val = pca.transform(X_val)
    if X_test.size > 0:
        X_test = pca.transform(X_test)

    print("PCA pre-processing finished.")

    return X_train, X_val, X_test

