import pickle
import random
import os
import numpy as np
from Preprocess import my_scaler
import tensorflow as tf
from librosa import display
import matplotlib.pyplot as plt

def get_batches(x, y, z, b_size, shuffle=True, seed=99):
    np.random.seed(seed)
    indxs = np.arange(tf.shape(x)[0])
    if shuffle:
        np.random.shuffle(indxs)

    def divide_chunks(l, n):
        # looping until length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    x_b, y_b, z_b = [], [], []
    indxs = divide_chunks(indxs, b_size)

    for indx_batch in indxs:
        # if len(indx_batch) != b_size:
        #     continue
        x_b.append(x[indx_batch])
        y_b.append(y[indx_batch])
        z_b.append(z[indx_batch])

    return x_b, y_b, z_b

def get_data(data_dir, w_length, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    file_data = open(os.path.normpath('/'.join([data_dir, 'TubeTech_data.pickle'])), 'rb')
    Z = pickle.load(file_data)
    inp = Z['inp']
    tars = Z['tar']
    attacks = Z['ratio']
    releases = Z['release']
    ratios = Z['ratio']
    thresholds = Z['threshold']
    gains = Z['gain']

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (-1, 1)
    # -----------------------------------------------------------------------------------------------------------------
    Z = [inp, tars]
    Z = np.array(Z)
    scaler = my_scaler(feature_range=(-1, 1))
    scaler.fit(Z)

    inp = scaler.transform(inp)
    tars = scaler.transform(tars)

    # scaler params?

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------
    del Z

    #divide chuncks
    L = 480000 // 2
    for i in range(tars.shape[0]):


    tars, notes, vels = get_batches(tars, attacks, thresholds, 1)


    return x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs


if __name__ == '__main__':
    data_dir = '../Files'
    w1 = 1
    w2 = 2
    w16 = 16
    x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, w_length=w2, seed=422)

    data = {'x': x, 'y': y, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test, 'scaler': scaler,
            'zero_value': zero_value, 'fs': fs}

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w1_[-1,1].pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()