import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle
import audio_format
from librosa import display
import matplotlib.pyplot as plt



def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def data_preparation():
    data_dir = 'C:/Users/riccarsi/Desktop/TubeTech2/Wav'
    factor = 2#6#3
    save_dir = '../Files'
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))
    L = 21120000
    collector = {'attack': [], 'release': [], 'ratio': [], 'threshold': [], 'gain': [], 'target': [], 'input': []}
    first = True
    amp = match_amplitude()
    for file in file_dirs:

        filename = os.path.split(file)[-1]
        metadata = filename.split('_', 10)
        attack = np.array(metadata[2], dtype=np.float32)
        release = np.array(metadata[4], dtype=np.float32)
        ratio = np.array(metadata[6], dtype=np.float32)
        threshold = np.array(metadata[8], dtype=np.float32)
        gain = np.array(metadata[-1].replace('.wav', ''), dtype=np.float32)

        fs, audio_stereo = wavfile.read(file) #fs= 96,000 Hz
        if first == True:
            inp = audio_stereo[960:L, 0]
            inp = audio_format.pcm2float(inp)
            inp = amp*inp
            inp = signal.resample_poly(inp, 1, factor)
            collector['input'].append(inp)
            first = False
        #960 due to 20ms of delay
        tar = audio_stereo[960+1:L+1, 1]
        tar = audio_format.pcm2float(tar)
        tar = signal.resample_poly(tar, 1, factor)

        #target is delayed by one sample due the system processing so
        #need to be moved
        #tar = tar[1:len(tar)]

        collector['target'].append(tar)
        collector['attack'].append(attack)
        collector['release'].append(release)
        collector['ratio'].append(ratio)
        collector['threshold'].append(threshold)
        collector['gain'].append(gain)
        #collector['index'].append(index)
        #index = index + 1

    file_data = open(os.path.normpath('/'.join([save_dir,'TubeTech_data.pickle'])), 'wb')
    pickle.dump(collector, file_data)
    file_data.close()


def divede_chuncks():
    data_dir = '../Files'
    file_data = open(os.path.normpath('/'.join([data_dir, 'TubeTech_data.pickle'])), 'rb')
    Z = pickle.load(file_data)

    inp = np.array(Z['input'])
    tars = np.array(Z['target'])
    attacks = np.array(Z['ratio'])
    releases = np.array(Z['release'])
    ratios = np.array(Z['ratio'])
    thresholds = np.array(Z['threshold'])
    gains = np.array(Z['gain'])
    collector = {'attack': [], 'release': [], 'ratio': [], 'threshold': [], 'gain': [], 'target': [], 'input': []}
    #divide chuncks
    L = 48000*10# - 1000 #// 2
    for n in range(tars.shape[0]):
        for i in range(tars.shape[1]//L):
            tar_temp = tars[n, i*L: (i+1)*L]

            #fig, ax = plt.subplots()
            #ax.set(ylim=[-0.3, 0.3])
            #display.waveshow(inp_temp, sr=48000, ax=ax)
            #plt.show()

            collector['target'].append(tar_temp)
            collector['attack'].append(attacks[n])
            collector['release'].append(releases[n])
            collector['ratio'].append(ratios[n])
            collector['threshold'].append(thresholds[n])
            collector['gain'].append(gains[n])
    for i in range(inp.shape[1] // L):

        inp_temp = inp[0, i * L: (i + 1) * L]
        collector['input'].append(inp_temp)

    file_data = open(os.path.normpath('/'.join([data_dir, 'TubeTech_data_chuncks.pickle'])), 'wb')
    pickle.dump(collector, file_data)
    file_data.close()

def match_amplitude():
    data_dir = 'C:/Users/riccarsi/Desktop/TubeTech2/Wav/Reference'

    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'TubeTech_difference_sines.wav'])))
    for file in file_dirs:
        fs, audio_stereo = wavfile.read(file) #fs= 96,000 Hz
        inp = audio_stereo[:, 0]
        tar = audio_stereo[:, 1]
        inp = audio_format.pcm2float(inp)
        tar = audio_format.pcm2float(tar)
        max_inp = np.max(inp)
        max_tar = np.max(tar)

        # inp = inp*max_tar/max_inp
        # max_inp = np.max(inp)
        # max_tar = np.max(tar)
        # max_inp == max_tar
        return max_tar/max_inp

if __name__ == '__main__':

    data_preparation()
    divede_chuncks()
