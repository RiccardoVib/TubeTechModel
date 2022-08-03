import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from TrainFunctionality import combinedLoss
# from tensorflow.python.keras.layers import MultiHeadAttention
from Preprocess import positional_encoding
from tensorflow.keras import layers
import numpy as np
import os
from GetDataTubeTech import get_data, get_test_data, get_scaler
import pickle
from scipy.io import wavfile


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def trainMultiAttention(data_dir, epochs, seed=422, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    ff_dim = kwargs.get('ff_dim', 512)
    num_heads = kwargs.get('num_heads', 8)
    d_model = kwargs.get('d_model', 512)
    model_save_dir = kwargs.get('model_save_dir', '/scratch/users/riccarsi/TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    loss_type = kwargs.get('loss_type', 'mse')
    w_length = kwargs.get('w_length', 16)
    generate_wav = kwargs.get('generate_wav', None)
    type = kwargs.get('type', 'int')
    inference = kwargs.get('inference', False)
    T = w_length
    D = 5

    # inputs layers
    inp_enc = tf.keras.Input(shape=(T, D))
    # inp_dec = tf.keras.Input(shape=[None, T, D])
    positional_encoding_enc = positional_encoding(T, d_model)
    # positional_encoding_dec = positional_encoding(T, d_model)
    inp_ = tf.keras.layers.Dense(d_model)(inp_enc)  # embedding
    # inp_dec = tf.keras.layers.Dense(d_model)(inp_dec) #embedding
    outputs_enc = inp_ + positional_encoding_enc
    # outputs_dec = inp_dec + positional_encoding_dec
    outputs_enc = TransformerBlock(d_model, num_heads, ff_dim)(outputs_enc)
    outputs_enc = tf.keras.layers.Dense(1, activation='sigmoid')(outputs_enc)

    model = Model(inputs=inp_enc, outputs=outputs_enc)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', metrics=['mse'], optimizer=opt)



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
        number_of_iterations = 50  # 7

        for n_iteration in range(number_of_iterations):
            x, y, x_val, y_val, scaler = get_data(data_dir=data_dir, window=w_length, index=n_iteration,
                                                  number_of_iterations=number_of_iterations, type=type, seed=seed)

            results = model.fit(x[:, :, :], y[:, -1], batch_size=b_size, epochs=epochs, verbose=0,
                                validation_data=(x_val[:, :, :], y_val[:, -1]),
                                callbacks=callbacks)
            results = {
                'Min_val_loss': np.min(results.history['val_loss']),
                'Min_train_loss': np.min(results.history['loss']),
                'b_size': b_size,
                'learning_rate': learning_rate,
                'drop': drop,
                'opt_type': opt_type,
                'loss_type': loss_type,
                'd_model': d_model,
                'ff_dim': ff_dim,
                'num_heads': num_heads,
                'w_length': w_length,
                'type': type,
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
    test_loss = model.evaluate(x_test[:, :, :], y_test[:, -1], batch_size=b_size, verbose=0)
    print('Test Loss: ', test_loss)
    scaler = get_scaler(data_dir=data_dir, type=type, seed=seed)

    if generate_wav is not None:
        predictions = model.predict(x_test[:, :, :], batch_size=b_size)
        predictions = (scaler[0].inverse_transform(predictions[:, 0, 0])).reshape(-1)
        x_test = (scaler[0].inverse_transform(x_test[:, -1, 0])).reshape(-1)
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
        'd_model': d_model,
        'ff_dim': ff_dim,
        'num_heads': num_heads,
        'w_length': w_length,
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
    data_dir = '../../Files'
    seed = 422

    trainMultiAttention(data_dir=data_dir,
                        model_save_dir='../TrainedModels',
                        save_folder='MultiAttention',
                        ckpt_flag=True,
                        b_size=128,
                        learning_rate=0.001,
                        d_model=512,
                        ff_dim=512,
                        num_heads=8,
                        epochs=100,
                        loss_type='combined',
                        generate_wav=10,
                        w_length=16,
                        type='float',
                        inference=True)