__author__ = 'yihanjiang'
'''
bcjr_rnn_train.py: train a BCJR-like RNN for Turbo Decoder.
'''
import numpy as np

from utils import corrupt_signal, snr_db2sigma, get_test_sigmas, conv_enc
import commpy.channelcoding.convcode as cc
from commpy.utilities import hamming_dist

import multiprocessing as mp

###
# from conv_decoder import get_test_model
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
from keras.callbacks import LearningRateScheduler
from keras.layers.convolutional import Conv1D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
import keras
from keras.layers import Input, Embedding, LSTM,GRU, Dense, TimeDistributed, Lambda
from keras.models import Model
from keras.layers.wrappers import  Bidirectional

from keras.legacy import interfaces
from keras.optimizers import Optimizer

import keras.backend as K

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.backend.tensorflow_backend import set_session

from utils import get_test_sigmas, errors, snr_db2sigma, conv_enc
import time
import copy
    


def benchmark_compute(args, model_decoder, PARAMS={'noise_sigma': 0, 'trellis1': 0, 'M': 0, 'tb_depth': 0}, isConvDecode=False):
    print("RUNNING BENCH COMPUTE")
    """
    Computes model_decoder for one test_snr value
    """
    np.random.seed()

    if not isConvDecode:

        # Create coded message
        message_bits = np.random.randint(0, 2, args.block_len)
        coded_bits = cc.conv_encode(message_bits, PARAMS['trellis1'])
        received  = corrupt_signal(coded_bits, noise_type =args.noise_type, sigma = PARAMS['noise_sigma'])

        # make fair comparison between (100, 204) convolutional code and (100,200) RNN decoder, set the additional bit to 0
        received[-2*int(PARAMS['M']):] = 0.0
        
        # decode bits
        decoded_bits = model_decoder(args, received, PARAMS)
    
        # Process the decoded bits and return final error
        decoded_bits = decoded_bits[:-int(PARAMS['M'])]

    else:
        message_bits  = np.random.randint(0,2,(1,args.block_len,1))
        received = 2.0*conv_enc(message_bits, args)  - 1.0
        
        # decode bits
        decoded_bits = model_decoder(args, received, PARAMS)

    num_bit_errors = hamming_dist(message_bits.flatten(), decoded_bits.flatten())
    return num_bit_errors


def single_simulation_onearg(onearg):
    # TODO: pool fails due to model_test
    args, noise_sigma, trellis1, M, decoder_obj, dec_weight, model_test = onearg
    return benchmark_compute(args, decoder_obj, PARAMS = {'noise_sigma': noise_sigma, 'trellis1': trellis1, 'M': M, 'tb_depth': 15, 'dec_weight': dec_weight, 'model_test': model_test}, isConvDecode=model_test != 0)


def bench_runner(args, decoder_obj, simulation_option=0, dec_weight='', get_test_model=0):
    """
    Runs test simulation of a provided decoding algorithm
    
    REQUIRED Args
    ---------
        M, enc1, enc2, feedback
        snr_test_start, snr_test_end, snr_points
        num_cpu
        num_block_err, batch_size, block_len, noise_type
    """
    M = np.array([args.M])
    generator_matrix = np.array([[args.enc1,args.enc2]])
    feedback = args.feedback
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)

    SNRS, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)
    
    nb_errors          = np.zeros(test_sigmas.shape)
    nb_block_errors    = np.zeros(test_sigmas.shape)

    for idx, noise_sigma in enumerate(test_sigmas):
        print('[testing]SNR: %4.1f'% SNRS[idx])
        
        num_block_test = 0
        pool = mp.Pool(processes=args.num_cpu)

        model_test = get_test_model(args, dec_weight, noise_sigma) if not get_test_model==0 else 0
        while nb_block_errors[idx] < args.num_block_err: #100)
            num_block_test += args.batch_size if not simulation_option == 1 else 1

            if simulation_option == 0:
                if not get_test_model == 0:
                    print("ERROR: CANNOT RUN CONV_DECODE WITH THREADS. RUNNING SIMULATION_OPTION (1).")
                    simulation_option = 1
                else:
                    # multithreading option
                    # TODO: FAILING BECAUSE OF [model_test] IN ONEARG. 
                    # One solution: use copy.deepcopy(onearg); however, there is a problem with copying model_test multiple times
                    onearg=(args, noise_sigma, trellis1, M, decoder_obj, dec_weight, model_test)
                    oneargv=[onearg for i in range(args.batch_size)] # just repeat it batch_size times
                    results1 = pool.map(single_simulation_onearg, oneargv)
            if simulation_option == 1:
                # single simulation option
                results1 = [benchmark_compute(args, decoder_obj, PARAMS = {'noise_sigma': noise_sigma, 'trellis1': trellis1, 'M': M, 'tb_depth': 15, 'dec_weight': dec_weight, 'model_test': model_test}, isConvDecode=(model_test!=0))]
            elif simulation_option == 2:
                # run batch of simulations option
                results1 = [
                    benchmark_compute(args, decoder_obj, PARAMS = {'noise_sigma': noise_sigma, 'trellis1': trellis1, 'M': M, 'tb_depth': 15, 'dec_weight': dec_weight, 'model_test': model_test}, isConvDecode=(model_test!=0))
                    for i in range(args.batch_size)
                ]

            nb_errors[idx] += sum(results1)
            nb_block_errors[idx] += sum(np.array(results1)>0)

            if num_block_test % 20 == 0: # print intermeadiate results every so often
                print('%8d %8d %8d'% (num_block_test, int(nb_block_errors[idx]), nb_errors[idx]))


        nb_errors[idx] /= float(args.block_len*num_block_test)
        nb_block_errors[idx] /= float(num_block_test)
        
        pool.close()
        print('%8d %8d %8d'% (num_block_test, int(nb_block_errors[idx]), nb_errors[idx]))
        print('[Testing]BER ', nb_errors)
        print('[Testing]BLER ', nb_block_errors)
    
    
    print('\n[Result]SNR:', SNRS)
    print('[Result]BER ', nb_errors)
    print('[Result]BLER ', nb_block_errors)
    return nb_errors, nb_block_errors









