__author__ = 'yihanjiang'
'''
bcjr_rnn_train.py: train a BCJR-like RNN for Turbo Decoder.
'''
from keras import backend as K
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda

from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional
from keras import regularizers

import sys
import pickle
import numpy as np
import math

from bcjr_util import generate_bcjr_example
from conv_decoder import build_decoder, errors

import commpy.channelcoding.interleavers as RandInterlv
import commpy.channelcoding.convcode as cc
import commpy.channelcoding.turbo as turbo
from utils import corrupt_signal, snr_db2sigma
from commpy.utilities import hamming_dist

def get_test_sigmas(snr_start, snr_end, snr_points):
    SNR_dB_start_Eb = snr_start
    SNR_dB_stop_Eb = snr_end
    SNR_points = snr_points
    if (SNR_points == 1):
        snr_interval = 0
    else:
        snr_interval = (SNR_dB_stop_Eb - SNR_dB_start_Eb)* 1.0 /  (SNR_points-1)
    SNRS_dB = [snr_interval* item + SNR_dB_start_Eb for item in range(SNR_points)]
    SNRS_dB_Es = [item + 10*np.log10(1.0/2.0) for item in SNRS_dB]
    test_sigmas = np.array([np.sqrt(1/(2*10**(float(item)/float(10)))) for item in SNRS_dB_Es])

    SNRS = SNRS_dB
    print('[testing] SNR range in dB ', SNRS)

    return SNRS, test_sigmas

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_block_train', type=int, default=100)
    parser.add_argument('-num_block_test', type=int, default=100)
    parser.add_argument('-block_len', type=int, default=10000)
    parser.add_argument('-num_dec_iteration', type=int, default=6)
    parser.add_argument('-code_rate', type=int, default=2)
    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)
    parser.add_argument('-M',  type=int, default=2, help="Number of delay elements in the convolutional encoder")
    parser.add_argument('-num_cpu', type=int, default=4)
    parser.add_argument('-snr_test_start', type=float, default=0.0)
    parser.add_argument('-snr_test_end', type=float, default=7.0)
    parser.add_argument('-snr_points', type=int, default=8)
    parser.add_argument('-init_nw_model', type=str, default='default')
    parser.add_argument('-rnn_setup', choices = ['lstm', 'gru'], default = 'lstm')
    parser.add_argument('-rnn_direction', choices = ['bd', 'sd'], default = 'bd')
    parser.add_argument('-num_Dec_layer', type=int, default=2)
    parser.add_argument('-num_Dec_unit', type=int, default=200)
    parser.add_argument('-batch_size',  type=int, default=10)
    parser.add_argument('-learning_rate',  type=float, default=0.001)
    parser.add_argument('-num_epoch',  type=int, default=20)

    parser.add_argument('-noise_type', choices = ['awgn', 't-dist','hyeji_bursty'], default='awgn')
    parser.add_argument('-train_snr', type=float, default=-1.0)
    parser.add_argument('-loss', choices = ['binary_crossentropy', 'mse', 'mae'], default='mse')
    parser.add_argument('-train_channel_low', type=float, default=0.0)
    parser.add_argument('-train_channel_high', type=float, default=8.0)
    parser.add_argument('-radar_power', type=float, default=20.0)
    parser.add_argument('-radar_prob', type=float, default=0.05)
    parser.add_argument('-fixed_var', type=float, default=0.00)
    parser.add_argument('--GPU_proportion', type=float, default=1.00)
    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    args = parser.parse_args()

    print(args)

    print('[ID]', args.id)

    return args



def test_bcjr_ber(args, model_path):
    '''
    under construction. ugly code available via requirement. ETA 0228.
    '''
    pass

def train_decoder(args):
    """
    Trains a BCJR-like LSTRM decoder for turbo codes. See Appendix B of [1] and Figures 12-14.
    """
    print('[BCJR Setting Parameters] Network starting path is ',                 args.init_nw_model)
    print('[BCJR Setting Parameters] Initial learning_rate is ',                 args.learning_rate)
    print('[BCJR Setting Parameters] Training batch_size is ',                   args.batch_size)
    print('[BCJR Setting Parameters] Training num_epoch is ',                    args.num_epoch)
    print('[BCJR Setting Parameters] Turbo Decoding Iteration ',                 args.num_dec_iteration)
    print('[BCJR Setting Parameters] RNN Direction is ', args.rnn_direction)
    print('[BCJR Setting Parameters] RNN Model Type is ', args.rnn_setup)
    print('[BCJR Setting Parameters] Number of RNN layer is ', args.num_Dec_layer)
    print('[BCJR Setting Parameters] Number of RNN unit is ', args.num_Dec_unit)
    M = np.array([args.M])
    generator_matrix = np.array([[args.enc1,args.enc2]])
    feedback = args.feedback
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    interleaver = RandInterlv.RandInterlv(args.block_len, 0)
    p_array = interleaver.p_array
    print('[BCJR Code Codec] Encoder', 'M ', M, ' Generator Matrix ', generator_matrix, ' Feedback ', feedback)
    codec  = [trellis1, trellis2, interleaver]
    print('[BCJR Setting Parameters] Training Data SNR is ', args.train_snr, ' dB')
    print('[BCJR Setting Parameters] Code Block Length is ', args.block_len)
    print('[BCJR Setting Parameters] Number of Train Block is ', args.num_block_train, ' Test Block ', args.num_block_test)
    model = build_decoder(args)
    bcjr_inputs_train, bcjr_outputs_train = generate_bcjr_example(args.num_block_train, args.block_len,
                                                                  codec, is_save = False,num_iteration = args.num_dec_iteration,
                                                                  train_snr_db = args.train_snr, save_path = './tmp/')

    bcjr_inputs_test,  bcjr_outputs_test  = generate_bcjr_example(args.num_block_test, args.block_len,
                                                                  codec, is_save = False, num_iteration = args.num_dec_iteration,
                                                                  train_snr_db = args.train_snr, save_path = './tmp/')

    train_batch_size  = args.batch_size                   # 100 good.
    test_batch_size   = args.batch_size
    input_feature_num = 3

    optimizer= keras.optimizers.adam(lr=args.learning_rate)
    model.compile(optimizer=optimizer,loss=args.loss, metrics=['mae'])



    if args.init_nw_model != 'default':
        model.load_weights(args.init_nw_model)
        print('[BCJR][Warning] Loaded Some init weight', args.init_nw_model)
    else:
        print('[BCJR][Warning] Train from scratch, not loading weight!')

    model.fit(x=bcjr_inputs_train, y=bcjr_outputs_train, batch_size=train_batch_size,
              epochs=args.num_epoch,  validation_data= (bcjr_inputs_test, bcjr_outputs_test))
    model.save_weights('./tmp/bcjr_train'+args.id +'_1.h5')
    print('[BCJR] Saved Model at', './tmp/bcjr_train'+args.id +'_1.h5')
    test_bcjr_ber(args,'./tmp/bcjr_train'+args.id +'_1.h5' )

def bcjr_bench(args, num_iteration=6, max_batch=4, batch_size=100):
    """
    Benchmark for MAP algorithm in Figure 3 of [1]. Under construction by Josh as of 6/11/2020.
    """

    """
    tb_depth = 15
    np.random.seed()
    message_bits = np.random.randint(0, 2, args.block_len)
    coded_bits = cc.conv_encode(message_bits, trellis1)

    received  = corrupt_signal(coded_bits, noise_type =args.noise_type, sigma = test_sigmas[idx],
                               vv =args.v, radar_power = args.radar_power, radar_prob = args.radar_prob,
                               denoise_thd = args.radar_denoise_thd)
    # make fair comparison between (100, 204) convolutional code and (100,200) RNN decoder, set the additional bit to 0
    received[-2*int(M):] = 0.0
    decoded_bits = cc.viterbi_decode(received.astype(float), trellis1, tb_depth, decoding_type='unquantized')
    decoded_bits = decoded_bits[:-int(M)]
    num_bit_errors = hamming_dist(message_bits, decoded_bits)
    return num_bit_errors
    """
    # SET UP BCJR ENCODING STRUCTURE
    M = np.array([args.M])
    generator_matrix = np.array([[args.enc1,args.enc2]])
    feedback = args.feedback
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)
    trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)
    
    #interleaver = RandInterlv.RandInterlv(args.block_len, 0)
    
    #p_array = interleaver.p_array
    # Initialize BCJR input/output Pair for training (Is that necessary?)
    input_feature_num = 3
    #bcjr_inputs  = np.zeros([2*num_iteration, num_block, block_len ,input_feature_num])
    #bcjr_outputs = np.zeros([2*num_iteration, num_block, block_len ,1        ])
    # Generate Noisy Input For Turbo Decoding
    nb_errors = np.zeros(args.snr_points)
    
    SNRS, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)
    
    for idx, noise_sigma in enumerate(test_sigmas):
        num_batch = 0
        print("Testing snr:", SNRS[idx])
        for _ in range(max_batch):
            if (nb_errors[idx]>100):
                break
            num_batch += 1
            print("Batch:", num_batch)
            for block in range(batch_size):
                message_bits = np.random.randint(0, 2, args.block_len)
                
                ans = cc.conv_encode(message_bits, trellis1)
                ans = ans[:2*args.block_len]
                ans = ans.reshape((args.block_len, 2))
                sys = ans[:, 0]
                par = ans[:, 1]
                #print("[[MESSAGE_BITS]]: ", message_bits)
                #print("[[UNCORRUPTED_PAR]]: ", par)
            
                sys_symbols       = corrupt_signal(sys, noise_type =args.noise_type, sigma = noise_sigma)
                non_sys_symbols   = corrupt_signal(par, noise_type =args.noise_type, sigma = noise_sigma)
                # Use the Commpy BCJR decoding algorithm
                noise_variance = noise_sigma**2
                L_int = np.zeros(len(sys_symbols))
                L_ext = L_int
                
                for iter in range(num_iteration):
                    L_int = L_ext
                    [L_ext_1, decoded_bits] = turbo.map_decode(sys_symbols, non_sys_symbols,
                                                        trellis1, noise_variance, L_int, 'decode')
                    L_ext = L_ext_1 - L_int
                    
                #print(L_ext)
                
                #print("[[SYS_SYMBOLS]]: ", sys_symbols)
                #print("[[NON_SYS_SYMBOLS]]: ", non_sys_symbols)
            
                #print("[[DECODED_BITS]]: ", decoded_bits)
                num_bit_errors = hamming_dist(message_bits, decoded_bits)
                nb_errors[idx] += num_bit_errors
            
        nb_errors[idx] /= (args.block_len*num_batch*batch_size)
    print(nb_errors)
    return nb_errors

if __name__ == '__main__':
    args = get_args()
    #train_decoder(args)
    bcjr_bench(args)



