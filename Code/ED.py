# import tensorboard
# load_ext tensorboard
# rm -rf ./logs/
import datetime

import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from TrainFunctionality import combinedLoss
from GetDataTubeTech import get_data, get_test_data, get_scaler
from scipy.io import wavfile
from scipy import signal
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import pickle
import numpy as np
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def trainED(data_dir, epochs, seed=422, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    encoder_units = kwargs.get('encoder_units', [64])
    decoder_units = kwargs.get('decoder_units', [64])
    if encoder_units[-1] != decoder_units[0]:
        raise ValueError('Final encoder layer must same units as first decoder layer!')
    model_save_dir = kwargs.get('model_save_dir', '/scratch/users/riccarsi/TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    loss_type = kwargs.get('loss_type', 'mse')
    w_length = kwargs.get('w_length', 16)
    generate_wav = kwargs.get('generate_wav', None)
    type = kwargs.get('type', 'int')
    inference = kwargs.get('inference', False)

    layers_enc = len(encoder_units)
    layers_dec = len(decoder_units)
    n_units_enc = ''
    for unit in encoder_units:
        n_units_enc += str(unit) + ', '

    n_units_dec = ''
    for unit in decoder_units:
        n_units_dec += str(unit) + ', '

    n_units_enc = n_units_enc[:-2]
    n_units_dec = n_units_dec[:-2]

    # x, y, x_val, y_val, scaler = get_data(data_dir=data_dir, window=w_length, index=0, number_of_iterations=7, seed=seed)

    # T past values used to predict the next value
    T = w_length  # x.shape[1] #time window
    D = 5  # x.shape[2] #features

    encoder_inputs = Input(shape=(T - 1, D), name='enc_input')
    first_unit_encoder = encoder_units.pop(0)
    if len(encoder_units) > 0:
        last_unit_encoder = encoder_units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(encoder_inputs)
        for i, unit in enumerate(encoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs, state_h, state_c = LSTM(last_unit_encoder, return_state=True, name='LSTM_EnFin')(outputs)
    else:
        outputs, state_h, state_c = LSTM(first_unit_encoder, return_state=True, name='LSTM_En')(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(1, 1), name='dec_input')
    first_unit_decoder = decoder_units.pop(0)
    if len(decoder_units) > 0:
        last_unit_decoder = decoder_units.pop()
        outputs = LSTM(first_unit_decoder, return_sequences=True, name='LSTM_De0', dropout=drop)(decoder_inputs,
                                                                                                 initial_state=encoder_states)
        for i, unit in enumerate(decoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_De' + str(i + 1), dropout=drop)(outputs)
        outputs, _, _ = LSTM(last_unit_decoder, return_sequences=True, return_state=True, name='LSTM_DeFin',
                             dropout=drop)(outputs)
    else:
        outputs, _, _ = LSTM(first_unit_decoder, return_sequences=True, return_state=True, name='LSTM_De',
                             dropout=drop)(
            decoder_inputs,
            initial_state=encoder_states)
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    decoder_outputs = Dense(1, activation='sigmoid', name='DenseLay')(outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()

    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Please pass opt_type as either Adam or SGD')

    if loss_type == 'mae':
        model.compile(loss='mae', metrics=['mae'], optimizer=opt)
    elif loss_type == 'mse':
        model.compile(loss='mse', metrics=['mse'], optimizer=opt)
    elif loss_type == 'combined':
        model.compile(loss=combinedLoss, metrics=combinedLoss, optimizer=opt)
    else:
        raise ValueError('Please pass loss_type as either MAE or MSE')

    scaler = get_scaler(data_dir=data_dir, type=type, seed=seed)

    callbacks = []
    if ckpt_flag:
        ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
        ckpt_path_latest = os.path.normpath(
            os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
        ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
        ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
            os.makedirs(os.path.dirname(ckpt_dir_latest))

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                           save_best_only=True, save_weights_only=True, verbose=1)
        ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss',
                                                                  mode='min',
                                                                  save_best_only=False, save_weights_only=True,
                                                                  verbose=1)
        callbacks += [ckpt_callback, ckpt_callback_latest]
        latest = tf.train.latest_checkpoint(ckpt_dir_latest)
        if latest is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(latest)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001, patience=20,
                                                               restore_best_weights=True, verbose=0)
    callbacks += [early_stopping_callback]
    if not inference:
        # train
        number_of_iterations = 50

        for n_iteration in range(number_of_iterations):
            print("Getting data")
            x, y, x_val, y_val, scaler = get_data(data_dir=data_dir, window=w_length, index=n_iteration,
                                                  number_of_iterations=number_of_iterations, type=type, seed=seed)

            print("Starting Training")
            results = model.fit([x[:, :-1, :], x[:, -1, 0].reshape(x.shape[0], 1, 1)], y[:, -1], batch_size=b_size, epochs=epochs, verbose=1,
                                validation_data=([x_val[:, :-1, :], x_val[:, -1, 0].reshape(x_val.shape[0], 1, 1)], y_val[:, -1]),
                                callbacks=callbacks)
            print(n_iteration)
            print("Traininng done")

            results = {
                'Min_val_loss': np.min(results.history['val_loss']),
                'Min_train_loss': np.min(results.history['loss']),
                'b_size': b_size,
                'learning_rate': learning_rate,
                'drop': drop,
                'opt_type': opt_type,
                'loss_type': loss_type,
                'layers_enc': layers_enc,
                'layers_dec': layers_dec,
                'n_units_enc': n_units_enc,
                'n_units_dec': n_units_dec,
                'w_length': w_length,
                # 'Train_loss': results.history['loss'],
                'Val_loss': results.history['val_loss']
            }
            # print(results)
            if ckpt_flag:
                with open(os.path.normpath(
                        '/'.join([model_save_dir, save_folder, 'results_it_' + str(n_iteration) + '.txt'])), 'w') as f:
                    for key, value in results.items():
                        print('\n', key, '  : ', value, file=f)
                    pickle.dump(results,
                                open(os.path.normpath(
                                    '/'.join([model_save_dir, save_folder, 'results_it_' + str(n_iteration) + '.pkl'])),
                                     'wb'))

    x_test, y_test = get_test_data(data_dir=data_dir, window=w_length, type=type, seed=seed)
    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
    test_loss = model.evaluate([x_test[:, :-1, :], x_test[:, -1, 0].reshape(-1, 1)], y_test[:, -1], batch_size=b_size,
                                   verbose=0)
    print('Test Loss: ', test_loss)

    if generate_wav is not None:
        predictions = model.predict([x_test[:, :-1, :], x_test[:, -1, 0].reshape(-1, 1)], batch_size=b_size)
        predictions = (scaler[0].inverse_transform(predictions)).reshape(-1)
        x_test = (scaler[0].inverse_transform(x_test[:, :, 0])).reshape(-1)
        y_test = (scaler[0].inverse_transform(y_test[:, -1])).reshape(-1)

        # Define directories
        pred_name = '_pred.wav'
        inp_name = '_inp.wav'
        tar_name = '_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Save Wav files
        predictions = predictions.astype('int16')
        x_test = x_test.astype('int16')
        y_test = y_test.astype('int16')
        wavfile.write(pred_dir, 48000, predictions)
        wavfile.write(inp_dir, 48000, x_test)
        wavfile.write(tar_dir, 48000, y_test)

    results = {
        'Test_Loss': test_loss,
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        'drop': drop,
        'opt_type': opt_type,
        'loss_type': loss_type,
        'layers_enc': layers_enc,
        'layers_dec': layers_dec,
        'n_units_enc': n_units_enc,
        'n_units_dec': n_units_dec,
        'w_length': w_length,
        'type': type,
        # 'Train_loss': results.history['loss'],
        'Val_loss': results.history['val_loss']
    }
    print(results)

    if ckpt_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    return results


if __name__ == '__main__':
    data_dir = '../Files'
    seed = 422
    # start = time.time()
    trainED(data_dir=data_dir,
            model_save_dir='../TrainedModels',
            save_folder='ED_TubeTech',
            ckpt_flag=True,
            b_size=128,
            learning_rate=0.001,
            encoder_units=[64],
            decoder_units=[64],
            epochs=100,
            loss_type='combined',
            generate_wav=1,
            w_length=16,
            type='int',
            inference=True)
    # end = time.time()
    # print(end - start)