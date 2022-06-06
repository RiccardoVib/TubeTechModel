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
        #x_b.append(x[indx_batch%27])
        #y_b.append(y[indx_batch])
        #z_b.append(z[indx_batch])

    return indexes

def get_data(data_dir, w_length, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    file_data = open(os.path.normpath('/'.join([data_dir, 'TubeTech_data_chuncks.pickle'])), 'rb')
    Z = pickle.load(file_data)
    inp = Z['input'][:30]
    tars = Z['target'][:30]
    attacks = Z['ratio'][:30]
    releases = Z['release'][:30]
    ratios = Z['ratio'][:30]
    thresholds = Z['threshold'][:30]
    gains = Z['gain'][:30]

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    Z = np.concatenate((tars, inp), axis=0)
    scaler = my_scaler(feature_range=(0, 1))
    scaler.fit(Z)

    inp = np.array(inp)
    tars = np.array(tars)
    #inp = scaler.transform(inp)
    #tars = scaler.transform(tars)

    # scaler params?
    del Z

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    indexes = get_batches(tars, 3)
    x_, y_, x, y, x_val, y_val, x_test, y_test = [], [], [], [], [], [], [], []

    N = tars.shape[0]
    n_train = N//100*70
    n_val = n_train + (N-n_train)//2
    n_test = n_train + n_val
    for ind in range(len(indexes)):
        for index in range(len(indexes[ind])):
            x_.append(inp[index%27])
            y_.append(tars[index])

            fig, ax = plt.subplots()
            #display.waveshow(np.array(x_)[index], sr=48000, ax=ax)#400
            display.waveshow(np.array(y_)[index], sr=48000, ax=ax)#0.3
            plt.show()

    x_ = np.array(x_)
    y_ = np.array(y_)
    x = x_[:n_train]
    y = y_[:n_train]
    x_val = x_[n_train:n_val]
    y_val = y_[n_train:n_val]
    x_test = x_[n_test:]
    y_test = y_[n_test:]

    return x, y, x_val, y_val, x_test, y_test, scaler

if __name__ == '__main__':
    data_dir = '../Files'
    w1 = 1
    w16 = 16
    x, y, x_val, y_val, x_test, y_test, scaler = get_data(data_dir=data_dir, w_length=w16, seed=422)

    data = {'x': x, 'y': y, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test, 'scaler': scaler}

    #file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared.pickle'])), 'wb')
    #pickle.dump(data, file_data)
    #file_data.close()