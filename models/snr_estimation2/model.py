## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

import models.snr_estimation as deepxi
from .gain import gfunc
from .network.selector import network_selector
from .inp_tgt import inp_tgt_selector
from .sig import InputTarget
from .utils import read_mat, read_wav, save_mat, save_wav
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input, Masking
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.lib.io import file_io
from tqdm import tqdm
import models.snr_estimation.se_batch as batch
import csv, math, os, pickle, random # collections, io, six
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import sys

sys.modules['deepxi'] = deepxi

# [1] Nicolson, A. and Paliwal, K.K., 2019. Deep learning for
#     minimum mean-square error approaches to speech enhancement.
#     Speech Communication, 111, pp.44-55.

class DeepXi():
    """
    Deep Xi model from [1].
    """
    def __init__(
        self,
        N_d,
        N_s,
        K,
        f_s,
        inp_tgt_type,
        network_type,
        min_snr,
        max_snr,
        snr_inter,
        sample_dir=None,
        ver='VERSION_NAME',
        train_s_list=None,
        train_d_list=None,
        sample_size=None,
        **kwargs
        ):
        """
        Argument/s
            N_d - window duration (samples).
            N_s - window shift (samples).
            K - number of frequency bins.
            f_s - sampling frequency.
            inp_tgt_type - input and target type.
            network_type - network type.
            min_snr - minimum SNR level for training.
            max_snr - maximum SNR level for training.
            stats_dir - path to save sample statistics.
            ver - version name.
            train_s_list - clean-speech training list to compute statistics.
            train_d_list - noise training list to compute statistics.
            sample_size - number of samples to compute the statistics from.
            kwargs - keyword arguments.
        """
        self.inp_tgt_type = inp_tgt_type
        self.network_type = network_type
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.snr_levels = list(range(self.min_snr, self.max_snr + 1, snr_inter))
        self.ver = ver
        self.train_s_list=train_s_list
        self.train_d_list=train_d_list
        
        inp_tgt_obj_path = kwargs['model_path'] + '/' + self.ver + '_inp_tgt.p'
        if os.path.exists(inp_tgt_obj_path):
            with open(inp_tgt_obj_path, 'rb') as f:
                self.inp_tgt = pickle.load(f)

        self.inp = Input(name='inp', shape=[None, self.inp_tgt.n_feat], dtype='float32')
        self.network = network_selector(self.network_type, self.inp,
            self.inp_tgt.n_outp, **kwargs)

        self.model = Model(inputs=self.inp, outputs=self.network.outp)

    def inferV2(
        self,
        test_x,
        test_x_len,
        test_epoch,
        model_path='model',
        out_type='y',
        gain='mmse-lsa',
        n_filters=40,
        saved_data_path=None
    ):
        if not isinstance(test_epoch, list): test_epoch = [test_epoch]
        if not isinstance(gain, list): gain = [gain]

        # The mel-scale filter bank is to compute an ideal binary mask (IBM)
        # estimate for log-spectral subband energies (LSSE).
        if out_type == 'subband_ibm_hat':
            mel_filter_bank = self.mel_filter_bank(n_filters)

        for e in test_epoch:
            if e < 1: raise ValueError("test_epoch must be greater than 0.")
            for g in gain:
                self.model.load_weights(model_path + '/epoch-' + str(e-1) + '/variables/variables')

                #print("Processing observations...")
                inp_batch, supplementary_batch, n_frames = self.observation_batch(test_x, test_x_len)

                #print("Performing inference...")
                tgt_hat_batch = self.model.predict(inp_batch, batch_size=1, verbose=0)

                #print("Saving outputs...")
                batch_size = len(test_x_len)
                output_data = []
                for i in range(batch_size):
                    inp = inp_batch[i,:n_frames[i],:]
                    tgt_hat = tgt_hat_batch[i,:n_frames[i],:]

                    # if tf.is_tensor(supplementary_batch):
                    supplementary = supplementary_batch[i,:n_frames[i],:]

                    #if saved_data_path is not None:
                    #   saved_data = read_mat(saved_data_path + '/' + base_name + '.mat')
                    #   supplementary = (supplementary, saved_data)
                    
                    if out_type == 'xi_hat':
                        xi_hat = self.inp_tgt.xi_hat(tgt_hat)
                        output_data.append(xi_hat)
                    elif out_type == 'gamma_hat':
                        gamma_hat = self.inp_tgt.gamma_hat(tgt_hat)
                        output_data.append(gamma_hat)
                    elif out_type == 's_STPS_hat':
                        s_STPS_hat = self.inp_tgt.s_stps_hat(tgt_hat)
                        output_data.append(s_STPS_hat)
                    elif out_type == 'y':
                        y = self.inp_tgt.enhanced_speech(inp, supplementary, tgt_hat, g).numpy()
                        output_data.append(y)
                    elif out_type == 'deepmmse':
                        xi_hat = self.inp_tgt.xi_hat(tgt_hat)
                        d_PSD_hat = np.multiply(np.square(inp), gfunc(xi_hat, xi_hat+1.0,
                            gtype='deepmmse'))
                        output_data.append(d_PSD_hat)
                    elif out_type == 'ibm_hat':
                        xi_hat = self.inp_tgt.xi_hat(tgt_hat)
                        ibm_hat = np.greater(xi_hat, 1.0).astype(bool)
                        output_data.append(ibm_hat)
                    elif out_type == 'subband_ibm_hat':
                        xi_hat = self.inp_tgt.xi_hat(tgt_hat)
                        xi_hat_subband = np.matmul(xi_hat, mel_filter_bank.transpose())
                        subband_ibm_hat = np.greater(xi_hat_subband, 1.0).astype(bool)
                        output_data.append(subband_ibm_hat)
                    elif out_type == 'cd_hat':
                        cd_hat = self.inp_tgt.cd_hat(tgt_hat)
                        output_data.append(cd_hat)
                    else: raise ValueError('Invalid output type.')
                return output_data

    def dataset(self, n_epochs, buffer_size=16):
        """
        Used to create a tf.data.Dataset for training.

        Argument/s:
            n_epochs - number of epochs to generate.
            buffer_size - number of mini-batches to keep in buffer.

        Returns:
            dataset - tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_generator(
            self.mbatch_gen,
            (tf.float32, tf.float32, tf.float32),
            (tf.TensorShape([None, None, self.inp_tgt.n_feat]),
                tf.TensorShape([None, None, self.inp_tgt.n_outp]),
                tf.TensorShape([None, None])),
            [tf.constant(n_epochs)]
            )
        dataset = dataset.prefetch(buffer_size)
        return dataset

    def observation_batch(self, x_batch, x_batch_len):
        """
        Computes observations (inp) from noisy speech recordings.

        Argument/s:
            x_batch - noisy-speech batch.
            x_batch_len - noisy-speech batch lengths.

        Returns:
            inp_batch - batch of observations (input to network).
            supplementary_batch - batch of noisy-speech short-time phase spectrums.
            n_frames_batch - number of frames in each observation.
        """
        batch_size = len(x_batch)
        max_n_frames = self.inp_tgt.n_frames(max(x_batch_len))
        inp_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_feat], np.float32)
        supplementary_batch = np.zeros([batch_size, max_n_frames, self.inp_tgt.n_feat], np.float32)
        n_frames_batch = [self.inp_tgt.n_frames(i) for i in x_batch_len]
        for i in range(batch_size):
            inp, supplementary = self.inp_tgt.observation(x_batch[i,:x_batch_len[i]])
            inp_batch[i,:n_frames_batch[i],:] = inp
            supplementary_batch[i,:n_frames_batch[i],:] = supplementary
        return inp_batch, supplementary_batch, n_frames_batch

class SaveWeights(Callback):  ### RENAME TO SaveModel
    """
    """
    def __init__(self, model_path):
        """
        """
        super(SaveWeights, self).__init__()
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        """
        """
        self.model.save(self.model_path + "/epoch-" + str(epoch))

class TransformerSchedular(LearningRateSchedule):
    """
    """
    def __init__(self, d_model, warmup_steps):
        """
        """
        super(TransformerSchedular, self).__init__()
        self.d_model = float(d_model)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        """
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        """
        """
        config = {'d_model': self.d_model, 'warmup_steps': self.warmup_steps}
        return config
