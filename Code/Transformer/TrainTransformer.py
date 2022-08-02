import os
import tensorflow as tf
import numpy as np
import pickle
from GetDataTubeTech import get_data, get_test_data

from scipy.io import wavfile
from Transformer import Transformer
from TrainFunctionality import CustomSchedule, PlotLossesSubPlots
from tensorflow.keras.utils import Progbar
from GetDataTubeTech import get_data

def train_RAMT(data_dir, epochs, seed=422, data=None, **kwargs):

    # -----------------------------------------------------------------------------------------------------------------
    # Set-up model, optimiser, lr_sched and losses:
    # -----------------------------------------------------------------------------------------------------------------

    model_save_dir = kwargs.get('model_save_dir', '../../LSTM_TrainedModels')
    save_folder = kwargs.get('save_folder', 'LSTM_enc_dec_Testing')

    generate_wav = kwargs.get('generate_wav', None)
    ckpt_flag = kwargs.get('ckpt_flag', False)
    opt_type = kwargs.get('opt_type', 'Adam')
    plot_progress = kwargs.get('plot_progress', True)

    learning_rate = kwargs.get('learning_rate', None)
    b_size = kwargs.get('b_size', 16)
    num_layers = kwargs.get('num_layers', 4)
    d_model = kwargs.get('d_model', 128)
    dff = kwargs.get('dff', 512)
    num_heads = kwargs.get('num_heads', 8)
    drop = kwargs.get('drop', .2)
    window = kwargs.get('window', 16)

    output_dim = window
    param=5
    
    inference_flag = kwargs.get('inference_flag', False)
    device_num = kwargs.get('device_num', None)
    loss_type = kwargs.get('loss_type', 'thres_log')

    # if device_num is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "device_num"

    if learning_rate is None:
        learning_rate = CustomSchedule(d_model=d_model, warmup_steps=4000)
    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Please pass opt_type as either Adam or SGD')

    transformer = Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,  # Hidden layer size of feedforward networks
                              input_vocab_size=None,  # Not relevant for ours as we don't use embedding
                              target_vocab_size=None,
                              pe_input=window,  # Max length for positional encoding input
                              pe_target=window,
                              output_dim=output_dim,
                              rate=drop)  # Dropout rate

    # loss_fn = loss_fn()#tf.keras.losses.MeanAbsoluteError()

    if loss_type == 'mae':
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    elif loss_type == 'mse':
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError('Please give a valid loss function type')

    #mae_fn = tf.keras.losses.MeanAbsoluteError()       # loss_fn(sampling_rate=8820, a=.5, b=.5)
    #mse_fn = tf.keras.losses.MeanSquaredError()
    # loss_fn = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    #val_loss_mae = tf.keras.metrics.Mean(name='val_loss_mae')
    #val_loss_mse = tf.keras.metrics.Mean(name='val_loss_mse')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    #test_loss_mae = tf.keras.metrics.Mean(name='test_loss_mae')
    #test_loss_mse = tf.keras.metrics.Mean(name='test_loss_mse')

    # -----------------------------------------------------------------------------------------------------------------
    # Define the training functionality
    # -----------------------------------------------------------------------------------------------------------------
    @tf.function
    def train_step(inp, tar):
        inp_enc = inp[:, :-1, :]
        inp_dec = inp[:, -1, 0]
        tar_real = tar[:, -1]
        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp_enc, inp_dec], training=True)

            # loss = loss_thres = loss_mae = loss_fn(tar_real, predictions)

            loss = loss_fn(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        opt.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss.update_state(loss)

    @tf.function
    def val_step(inp, tar, testing=False):
        inp_enc = inp[:, :-1, :]
        inp_dec = inp[:, -1, 0]
        tar_real = tar[:, -1]

        predictions, attn_weights = transformer([inp_enc, inp_dec], training=False)

        # loss = loss_thres = loss_mae = loss_fn(tar_real, predictions)
        loss = loss_fn(tar_real, predictions)
        #loss_mae = mae_fn(tar_real, predictions)
        #loss_mse = mse_fn(tar_real, predictions)

        if not testing:
            val_loss.update_state(loss)
            #val_loss_mae.update_state(loss_mae)
            #val_loss_mse.update_state(loss_mse)
        else:
            test_loss.update_state(loss)
            #test_loss_mae.update_state(loss_mae)
            #test_loss_mse.update_state(loss_mse)

        return attn_weights

    # -----------------------------------------------------------------------------------------------------------------
    # Set up checkpointing (saving) of the model (and load if present)
    # -----------------------------------------------------------------------------------------------------------------
    start_epoch = 0
    if ckpt_flag or inference_flag:
        #Z_shape = tf.shape(sigs)
        graph_signature = [
            #tf.TensorSpec((None, Z_shape[1], Z_shape[2]), tf.float32),
            #tf.TensorSpec((None, Z_shape[1], Z_shape[2]), tf.float32)
            tf.TensorSpec((None, window, param), tf.float32),
            tf.TensorSpec((None, 1, 1), tf.float32)
        ]

        @tf.function(input_signature=graph_signature)
        def inference(tar, inp):
            inp_enc = inp[:, :-1, :]
            inp_dec = inp[:, -1, 0]

            outputs = transformer([inp_enc, inp_dec], training=False)
            return outputs

        save_model_latest = os.path.normpath('/'.join([model_save_dir, save_folder, 'Latest']))
        save_model_best = os.path.normpath('/'.join([model_save_dir, save_folder, 'Best']))

        # Load model if dir exists
        if os.path.exists(save_model_latest):
            f = open('/'.join([os.path.dirname(save_model_best), 'epoch.txt']))
            ckpt_info = f.read()
            f.close()
            start_epoch = [int(s) for s in ckpt_info.split() if s.isdigit()][0]  # Get the latest epoch it trained
            print('Loading weights and starting from epoch ', start_epoch)
            if inference_flag:
                loaded = tf.saved_model.load(save_model_best)
            else:
                loaded = tf.saved_model.load(save_model_latest)

            # Need to make a single prediction for the model as it needs to compile:
            x_ = x_test[0]
            y_ = y_test[0]
            # Need to make a single prediction for the model as it needs to compile:
            transformer([tf.constant(x_.reshape(1, 15, 5), dtype='float32'),
                         tf.constant(y_.reshape(1, 1, 1), dtype='float32')],
                        training=False)

            for i in range(len(transformer.variables)):
                if transformer.variables[i].name != loaded.all_variables[i].name:
                    assert ValueError('Cannot load model, due to incompatible loaded and model...')
                transformer.variables[i].assign(loaded.all_variables[i].value())
        else:
            print('Weights were randomly initialised')

        # Try to load loss figure plot if exists
        if os.path.exists(os.path.normpath('/'.join([model_save_dir, save_folder, 'fig_progress.pickle']))) and plot_progress:
            try:
                fig_progress = pickle.load(
                    open(os.path.normpath('/'.join([model_save_dir, save_folder, 'fig_progress.pickle'])), 'rb'))
            except (ValueError, Exception):
                print('Could not load loss figure')
                pass

    ckpt_interval = 1

    # -----------------------------------------------------------------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------------------------------------------------------------
    _logs = [[], [], [], []]
    min_val_error = np.inf
    summary_res = 1
    if inference_flag:
        start_epoch = epochs
    for epoch in range(start_epoch, epochs):
        train_loss.reset_states()
        val_loss.reset_states()
        #val_loss_mae.reset_states()
        #val_loss_mse.reset_states()

        # Get batches
        x_batches, y_batches, x_val_batches, y_val_batches, scaler = get_data(data_dir, 1, 1, window, seed=422)
        # Set-up training progress bar
        n_batch = x_batches.shape[0]
        print("\nepoch {}/{}".format(epoch + 1, epochs))
        pb_i = Progbar(n_batch * b_size, stateful_metrics=['Loss: '])

        for batch_num in range(n_batch):
            x_batch = x_batches[batch_num].reshape(1, window, param)
            y_batch = y_batches[batch_num].reshape(1, window, 1)

            x_batch = tf.constant(x_batch, dtype='float32')
            y_batch = tf.constant(y_batch, dtype='float32')

            train_step(inp=x_batch, tar=y_batch)

            # Print progbar
            if batch_num % summary_res == 0:
                values = [('Loss: ', train_loss.result())]
                pb_i.add(b_size*summary_res, values=values)

        # -------------------------------------------------------------------------------------------------------------
        # Validate the model
        # -------------------------------------------------------------------------------------------------------------

        # Get batches
        #x_batches, y_batches = get_batches(x_val, y_val, b_size=b_size, shuffle=True, seed=epoch)
        n_batch = x_val_batches.shape[0]
        for batch_num in range(n_batch):
            x_batch = x_val_batches[batch_num].reshape(1, window, param)
            y_batch = y_val_batches[batch_num].reshape(1, window, 1)

            x_batch = tf.constant(x_batch, dtype='float32')
            y_batch = tf.constant(y_batch, dtype='float32')

            val_step(inp=x_batch, tar=y_batch)

        # Print validation losses:
        print('\nValidation Loss:', val_loss.result().numpy())
        #print('         MAE:   ', val_loss_mae.result().numpy())
        #print('         MSE:   ', val_loss_mse.result().numpy())

        # -------------------------
        # *** Checkpoint Model: ***
        # -------------------------
        if ckpt_flag:
            if (epoch % ckpt_interval == 0) or (val_loss.result() < min_val_error):
                to_save = tf.Module()
                to_save.inference = inference
                to_save.all_variables = list(transformer.variables)
                tf.saved_model.save(to_save, save_model_latest)

                if val_loss.result() < min_val_error:
                    print('*** New Best Model Saved to %s ***' % save_model_best)
                    to_save = tf.Module()
                    to_save.inference = inference
                    to_save.all_variables = list(transformer.variables)
                    tf.saved_model.save(to_save, save_model_best)
                    best_epoch = epoch

                epoch_dir = '/'.join([os.path.dirname(save_model_best), 'epoch.txt'])
                f = open(epoch_dir, 'w+')
                f.write('Latest Epoch: %s \nBest Epoch: %s \n' % (epoch + 1, best_epoch + 1))
                f.close()

        # -----------------------------
        # *** Plot Training Losses: ***
        # -----------------------------
        if plot_progress:
            if 'fig_progress' not in locals():
                # fig_progress = PlotLossesSame(epoch + 1,
                #                               Training=train_loss.result().numpy(),
                #                               Validation=val_loss.result().numpy(),
                #                               Val_Thres=val_loss_thres.result().numpy(),
                #                               Val_sThres=val_loss_sthres.result().numpy(),
                #                               Val_MAE=val_loss_mae.result().numpy(),
                #                               Val_MSE=val_loss_mse.result().numpy())
                fig_progress = PlotLossesSubPlots(epoch + 1,
                                                  Losses1={
                                                      'Training': train_loss.result().numpy(),
                                                      'Validation': val_loss.result().numpy(),
                                                  #},
                                                  #Losses2={
                                                  #    'Val_MAE': val_loss_mae.result().numpy(),
                                                  #    'Val_MSE': val_loss_mse.result().numpy()
                                                  })
            else:
                # fig_progress.on_epoch_end(Training=train_loss.result().numpy(),
                #                           Validation=val_loss.result().numpy(),
                #                           Val_Thres=val_loss_thres.result().numpy(),
                #                           Val_sThres=val_loss_sthres.result().numpy(),
                #                           Val_MAE=val_loss_mae.result().numpy(),
                #                           Val_MSE=val_loss_mse.result().numpy())
                fig_progress.on_epoch_end(
                    Losses1={
                        'Training': train_loss.result().numpy(),
                        'Validation': val_loss.result().numpy(),
                    #},
                    #Losses2={
                    #    'Val_MAE': val_loss_mae.result().numpy(),
                    #    'Val_MSE': val_loss_mse.result().numpy()
                    })

            # Store the plot if ckpting:
            if ckpt_flag:
                fig_progress.fig.savefig(os.path.normpath('/'.join([model_save_dir, save_folder, 'val_loss.png'])))
                if not os.path.exists(os.path.normpath('/'.join([model_save_dir, save_folder]))):
                    os.makedirs(os.path.normpath('/'.join([model_save_dir, save_folder])))
                pd.to_pickle(fig_progress, os.path.normpath('/'.join([model_save_dir, save_folder,
                                                                      'fig_progress.pickle'])))

        # Store currently best validation loss:
        if val_loss.result() < min_val_error:
            min_val_error = val_loss.result()

        # Append epoch losses to logs:
        _logs[0].append(train_loss.result().numpy())
        _logs[1].append(val_loss.result().numpy())
        #_logs[2].append(val_loss_mae.result().numpy())
        #_logs[3].append(val_loss_mse.result().numpy())

        if epoch == start_epoch:
            n_params = np.sum([np.prod(v.get_shape()) for v in transformer.variables])
            print('Number of parameters: ', n_params)

    # -----------------------------------------------------------------------------------------------------------------
    # Test the model
    # -----------------------------------------------------------------------------------------------------------------
    # # Load the best model:
    if ckpt_flag and not inference_flag:
        if os.path.exists(save_model_best):
            f = open('/'.join([os.path.dirname(save_model_best), 'epoch.txt']))
            ckpt_info = f.read()
            f.close()
            start_epoch = [int(s) for s in ckpt_info.split() if s.isdigit()][1]  # Get the latest epoch it trained
            print('Loading weights from best epoch ', start_epoch)
            loaded = tf.saved_model.load(save_model_best)

            x_test, y_test = get_data_test(data_dir, window, seed=422)
            x_ = x_test[0]
            y_ = y_test[0]
            # Need to make a single prediction for the model as it needs to compile:
            transformer([tf.constant(x_.reshape(1, window, param), dtype='float32'),
                         tf.constant(y_.reshape(1, window, 1), dtype='float32')],
                        training=False)
            for i in range(len(transformer.variables)):
                if transformer.variables[i].name != loaded.all_variables[i].name:
                    assert ValueError('Cannot load model, due to incompatible loaded and model...')
                transformer.variables[i].assign(loaded.all_variables[i].value())

    if 'n_params' not in locals():
        n_params = np.sum([np.prod(v.get_shape()) for v in transformer.variables])
    if 'epoch' not in locals():
        _logs = [[0]]*len(_logs)
        epoch = 0

    # Get batches
    n_batch = x_test.shape[0]
    for batch_num in range(n_batch):
        x_batch = x_test[batch_num].reshape(1, window, param)
        y_batch = y_test[batch_num].reshape(1, window, 1)

        x_batch = tf.constant(x_batch, dtype='float32')
        y_batch = tf.constant(y_batch, dtype='float32')

        val_step(inp=x_batch, tar=y_batch, testing=True)

    print('\n\nTest Loss: ', test_loss.result().numpy())
    #print('    MAE:   ', test_loss_mae.result().numpy())
    #print('    MSE:   ', test_loss_mse.result().numpy(), '\n\n')

    results = {
        'Test_Loss': test_loss.result().numpy(),
        #'Test_Loss_MAE': test_loss_mae.result().numpy(),
        #'Test_Loss_MSE': test_loss_mse.result().numpy(),
        'b_size': b_size,
        'loss_type': loss_type,
        'num_layers': num_layers,
        'd_model': d_model,
        'dff': dff,
        'num_heads': num_heads,
        'drop': drop,
        'n_params': n_params,
        'learning_rate': learning_rate if isinstance(learning_rate, float) else 'Sched',
        'min_val_loss': np.min(_logs[1]),
        #'min_val_MAE': np.min(_logs[2]),
        #'min_val_MSE': np.min(_logs[3]),
        'min_train_loss': np.min(_logs[0]),
        'val_loss': _logs[1],
        #'val_loss_mae': _logs[2],
        #'val_loss_mse': _logs[3],
        'train_loss': _logs[0],
    }

    if ckpt_flag and not inference_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    # -----------------------------------------------------------------------------------------------------------------
    # Save some Wav-file Predictions (from test set):
    # -----------------------------------------------------------------------------------------------------------------
    if generate_wav is not None:
        np.random.seed(333)
        gen_indxs = 0#np.random.choice(N, generate_wav)

        predictions, _ = transformer([
            tf.constant(x_test, dtype='float32'),
            tf.constant(y_test, dtype='float32')],
            training=False)
        predictions = predictions[:, :, 0].numpy()
        predictions = scaler[0].inverse_transform(predictions.T).reshape(-1)

        y_test = scaler[0].inverse_transform(y_test[:, -1]).reshape(-1)

        pred_name = 'Transf_pred.wav'
        tar_name = 'Transf_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Resample
        pred = predictions.astype('int16')
        tar = y_test.astype('int16')

        # Save Wav files
        wavfile.write(pred_dir, 44800, pred[0])
        wavfile.write(tar_dir, 44800, tar)
        
    return results


if __name__ == '__main__':
    data_dir = '../../Files'
    train_RAMT(
        data_dir=data_dir,
        model_save_dir=r'../TrainedModels',
        save_folder='Transformer_Testing_tube',
        ckpt_flag=True,
        plot_progress=False,
        loss_type='mse',
        window=16,
        b_size=128,
        learning_rate=0.0001,
        num_layers=3,
        d_model=64,
        dff=128,
        num_heads=2,
        drop=0.2,
        epochs=2000,
        seed=422,
        generate_wav=10,
        inference_flag=False)