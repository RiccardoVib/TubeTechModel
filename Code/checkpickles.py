import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle
import audio_format
from TrainFunctionality import STFT_loss_function


data_dir = '../Files'
file_data = open(os.path.normpath('/'.join([data_dir, 'Prepared_chuncks.pickle'])), 'rb')
Z = pickle.load(file_data)

y = np.array(Z['y'])

STFT_loss_function(y[1:4, 0:16], y[1:4, 0:16])