import pickle
import random
import os
import numpy as np
from Preprocess import my_scaler
import tensorflow as tf
from librosa import display
import matplotlib.pyplot as plt


def get_batches(x, b_size, shuffle=True, seed=99):
    np.random.seed(seed)
    indxs = np.arange(tf.shape(x)[0])
    if shuffle:
        np.random.shuffle(indxs)

    def divide_chunks(l, n):
        # looping until length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    indexes = []
    indxs = divide_chunks(indxs, b_size)

    for indx_batch in indxs:
        # if len(indx_batch) != b_size:
        #     continue
        indexes.append(indx_batch)
        # x_b.append(x[indx_batch%27])
        # y_b.append(y[indx_batch])
        # z_b.append(z[indx_batch])

    return indexes


def prepare_data(data_dir, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    file_data = open(os.path.normpath('/'.join([data_dir, 'TubeTech_data_chuncks.pickle'])), 'rb')
    Z = pickle.load(file_data)
    inp = Z['input']
    tars = Z['target']
    attacks = np.array(Z['ratio'])
    # releases = Z['release']
    ratios = np.array(Z['ratio'])
    thresholds = np.array(Z['threshold'])
    gains = np.array(Z['gain']) / 10

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    Z = np.concatenate((tars, inp), axis=0)
    scaler = my_scaler(feature_range=(0, 1))
    scaler.fit(Z)

    inp = np.array(inp)
    tars = np.array(tars)
    inp = scaler.transform(inp)
    tars = scaler.transform(tars)

    scaler_t = my_scaler(feature_range=(0, 1))
    scaler_t.fit(thresholds)
    thresholds = scaler.transform(thresholds)
    scaler_r = my_scaler(feature_range=(0, 1))
    scaler_r.fit(ratios)
    ratios = scaler.transform(ratios)
    scaler_a = my_scaler(feature_range=(0, 1))
    scaler_a.fit(attacks)
    attacks = scaler.transform(attacks)

    # scaler params?
    del Z

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    indexes = get_batches(tars, 1)
    x_, y_, x, y, x_val, y_val, x_test, y_test = [], [], [], [], [], [], [], []
    z_, z, z_val, z_test = [], [], [], []

    N = tars.shape[0]  # 1155
    n_train = N // 100 * 70  # 770
    n_val = n_train + (N - n_train) // 2  # 962
    N_validation = (N - n_train) // 2
    for ind in range(len(indexes)):
        for index in range(len(indexes[ind])):
            indice = indexes[ind][index]
            x_.append(inp[indice % 21])
            y_.append(tars[indice])
            z_.append([attacks[indice], ratios[indice], thresholds[indice], gains[indice]])

            # fig, ax = plt.subplots()
            # display.waveshow(inp[indexes[ind][index]%21], sr=48000, ax=ax)
            # display.waveshow(tars[indexes[ind][index]], sr=48000, ax=ax)
            # plt.show()

    x_ = np.array(x_)
    y_ = np.array(y_)
    z_ = np.array(z_)
    x = x_[:n_train]
    y = y_[:n_train]
    z = z_[:n_train]
    x_val = x_[n_train:n_val + 1]
    y_val = y_[n_train:n_val + 1]
    z_val = z_[n_train:n_val + 1]
    x_test = x_[n_val + 1:]
    y_test = y_[n_val + 1:]
    z_test = z_[n_val + 1:]

    return N, N_validation, x, y, z, x_val, y_val, z_val, x_test, y_test, z_test, scaler


def get_data(data_dir, index, number_of_iterations, window, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    file_data = open(os.path.normpath('/'.join([data_dir, 'Prepared_chuncks.pickle'])), 'rb')
    Z = pickle.load(file_data)

    scaler = Z['scaler']
    N = Z['N'] - 2 * Z['N_validation'] - 1
    n_iteration = N // number_of_iterations
    indeces = [int(n_iteration * index), int((1 + index) * n_iteration)]
    if indeces[1] >= N:
        indeces[1] = N - 1
    x = np.array(Z['x'][indeces[0]: indeces[1]])
    y = np.array(Z['y'][indeces[0]: indeces[1]])
    z = np.array(Z['z'][indeces[0]: indeces[1]])

    N_val = Z['N_validation']
    n_iteration_v = N_val // number_of_iterations
    indeces_v = [int(n_iteration_v * index), int((1 + index) * n_iteration_v)]

    if indeces_v[1] >= N_val:
        indeces_v[1] = N_val - 1

    x_val = np.array(Z['x_val'][indeces_v[0]: indeces_v[1]])
    y_val = np.array(Z['y_val'][indeces_v[0]: indeces_v[1]])
    z_val = np.array(Z['z_val'][indeces_v[0]: indeces_v[1]])

    all_inp, all_tar = [], []
    length = x.shape[1]
    n_examples = x.shape[0]
    for i in range(n_examples):
        for t in range(length - window):
            inp_temp = np.array([x[i, t:t + window], np.repeat(z[i, 0], window),
                                 np.repeat(z[i, 1], window), np.repeat(z[i, 2], window), np.repeat(z[i, 3], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(y[i, t:t + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    all_inp_val, all_tar_val = [], []
    n_examples = x_val.shape[0]
    for i in range(n_examples):
        for t in range(length - window):
            inp_temp = np.array([x_val[i, t:t + window], np.repeat(z_val[i, 0], window),
                                 np.repeat(z_val[i, 1], window), np.repeat(z_val[i, 2], window),
                                 np.repeat(z_val[i, 3], window)])
            all_inp_val.append(inp_temp.T)
            tar_temp = np.array(y_val[i, t:t + window])
            all_tar_val.append(tar_temp.T)

    all_inp_val = np.array(all_inp_val)
    all_tar_val = np.array(all_tar_val)

    return all_inp, all_tar, all_inp_val, all_tar_val, scaler


def get_test_data(data_dir, window, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    file_data = open(os.path.normpath('/'.join([data_dir, 'Prepared_chuncks.pickle'])), 'rb')
    Z = pickle.load(file_data)
    x = np.array(Z['x_test'])
    y = np.array(Z['y_test'])
    z = np.array(Z['z_test'])

    all_inp, all_tar = [], []
    length = x.shape[1]
    n_examples = x.shape[0]
    for i in range(n_examples):
        for t in range(length - window):
            inp_temp = np.array([x[i, t:t + window], np.repeat(z[i, 0], window),
                                 np.repeat(z[i, 1], window), np.repeat(z[i, 2], window), np.repeat(z[i, 3], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(y[i, t:t + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    return all_inp, all_tar


if __name__ == '__main__':
    data_dir = '../Files'
    N, N_validation, x, y, z, x_val, y_val, z_val, x_test, y_test, z_test, scaler = prepare_data(data_dir=data_dir,
                                                                                                 seed=422)

    data = {'N': N, 'N_validation': N_validation, 'x': x, 'y': y, 'z': z, 'x_val': x_val, 'y_val': y_val,
            'z_val': z_val, 'x_test': x_test, 'y_test': y_test, 'z_test': z_test, 'scaler': scaler}

    file_data = open(os.path.normpath('/'.join([data_dir, 'Prepared_chuncks.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()