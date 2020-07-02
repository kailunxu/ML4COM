"""
Test file to determine why conv_decoder.py was buggy (6/26/20), by Josh.
"""

import numpy as np
from utils import corrupt_signal, snr_db2sigma, get_test_sigmas, conv_enc
import commpy.channelcoding.convcode as cc
from commpy.utilities import hamming_dist
import multiprocessing as mp
from collections import defaultdict
import sys
from utils import get_test_sigmas, errors, snr_db2sigma, conv_enc
    
from run_benchmarks import get_test_model

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block_err', type=int, default=100)
    parser.add_argument('-num_block', type=int, default=12000)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-variation_block_len', type=int, default=100)
    parser.add_argument('-test_ratio',  type=int, default=10)

    parser.add_argument('-num_cpu',  type=int, default=1)

    parser.add_argument('-num_Dec_layer',  type=int, default=2)
    parser.add_argument('-num_Dec_unit',  type=int, default=200)

    parser.add_argument('-rnn_setup', choices = ['lstm', 'gru'], default = 'gru')

    parser.add_argument('-batch_size',  type=int, default=100) #200
    parser.add_argument('-learning_rate',  type=float, default=0.001)
    parser.add_argument('-num_epoch',  type=int, default=20)

    parser.add_argument('-code_rate',  type=int, default=2)

    parser.add_argument('-snr_test_start', type=float, default=-3.0)
    parser.add_argument('-snr_test_end', type=float, default=6.0)
    parser.add_argument('-snr_points', type=int, default=10)
    
    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)
    parser.add_argument('-M',  type=int, default=2, help="Number of delay elements in the convolutional encoder")

    parser.add_argument('-noise_type',        choices = ['awgn', 't-dist','hyeji_bursty'], default='awgn')
    parser.add_argument('-radar_power',       type=float, default=20.0)
    parser.add_argument('-radar_prob',        type=float, default=0.05)
    parser.add_argument('-radar_denoise_thd', type=float, default=10.0)
    parser.add_argument('-v',                 type=int,   default=3)
    
    parser.add_argument('-loss', choices = ['binary_crossentropy', 'mean_squared_error'], default = 'mean_squared_error')

    parser.add_argument('-train_channel_low', type=float, default=0.0)
    parser.add_argument('-train_channel_high', type=float, default=0.0)

    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    parser.add_argument('-Dec_weight', type=str, default='default')
    parser.add_argument('-type', choices=['train', 'test', 'test_batches', 'compareSnr', 'lengthExpand', 'bitTest', 'fig15'], default = 'compare snr')

    args = parser.parse_args()
    print(args)

    print('[ID]', args.id)
    return args
    
# np.set_printoptions(threshold=sys.maxsize)
args = get_args()
dec_weight='./tmp/conv_dec'+str(829033)+'.h5'

# VERSION RUN_BENCHMARKS
def run_bench_version(args):
    print("\nRUNNING BENCH VERSION")
    _, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)

    NB = []
    for idx, noise_sigma in enumerate(test_sigmas):
        print("IDX: ", idx)
        message_bits  = np.random.randint(0,2,(int(args.num_block/args.test_ratio),args.block_len,1))
        received = 2.0*conv_enc(message_bits, args)  - 1.0      
    
        model_test = get_test_model(args, dec_weight, noise_sigma)
        pd         = model_test.predict(received, verbose=0)
        decoded_bits = np.round(pd).astype(int)
    
        num_bit_errors = [hamming_dist(message_bits.flatten(), decoded_bits.flatten())]
        
        NB.append(sum(num_bit_errors))
        NB[-1] /= float(args.num_block/args.test_ratio)
        
        del model_test
        

    print("NB: ", NB)

# VERSION TEST
def run_test_version(args):
    print("\nRUNNING TEST VERSION")
    X_test_raw  = np.random.randint(0,2,int(args.num_block*args.block_len/args.test_ratio))
    X_test  = X_test_raw.reshape((int(args.num_block/args.test_ratio), args.block_len, 1))
    X_conv_test  = 2.0*conv_enc(X_test, args)  - 1.0

    SNRS_dB, _ = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)
    ber = []

    for idx, snr_db in enumerate(SNRS_dB):
        print("IDX: ", idx)
        model_test = get_test_model(args, dec_weight, snr_db)
        pd       = model_test.predict(X_conv_test, verbose=0)
        decoded_bits = np.round(pd)

        ber_err_rate  = sum(sum(sum(abs(decoded_bits-X_test))))*1.0/(X_test.shape[0] * X_test.shape[1])
        ber.append(ber_err_rate)
        del model_test

    print("BER: ", ber)

def run_adjusted_bench(args):
    print("\nRUNNING BENCH VERSION")
    SNRS_dB, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)

    print("SNRS_dB: ", SNRS_dB)
    print("test_sigmas: ", test_sigmas)

    
    NB = []
    ber = []
    # for idx, snr_db in enumerate(SNRS_dB): # CHANGE 1
    for idx, noise_sigma in enumerate(test_sigmas):
        print("IDX: ", idx, "\tNB:", NB, "\n\t\tBER: ", ber)
        
        message_bits  = np.random.randint(0,2,(int(args.num_block/args.test_ratio),args.block_len,1))
        received = 2.0*conv_enc(message_bits, args)  - 1.0      

        
        model_test = get_test_model(args, dec_weight, noise_sigma)
        pd         = model_test.predict(received, verbose=0)
        decoded_bits = np.round(pd).astype(int)
        del model_test


        ber_err_rate  = sum(sum(sum(abs(decoded_bits-message_bits))))*1.0/(message_bits.shape[0] * message_bits.shape[1])
        print("BER_ERR_RATE: ", ber_err_rate)

        y  = sum(sum(sum(abs(decoded_bits-message_bits))))
        print("Y: ", y)

        x = hamming_dist(decoded_bits.flatten(), message_bits.flatten())
        print("X: ", x/(args.num_block/args.test_ratio*args.block_len))
        
        # print("DECODED: ", decoded_bits.flatten())
        # print("MESSAGE: ", message_bits.flatten())
        # print()
        continue
        # raise NotImplementedError

        ber.append(ber_err_rate)
        
        num_bit_errors = [hamming_dist(message_bits.flatten(), decoded_bits.flatten())]
        NB.append(sum(num_bit_errors))
        NB[-1] /= float(args.num_block/args.test_ratio)
        
        

    print("NB: ", NB)
    print("BER: ", ber)


# run_bench_version(args)
# bench_old = [3.299166666666667, 3.7475, 4.1175, 4.6275, 4.968333333333334, 5.23, 5.465, 5.935, 6.206666666666667, 6.4825]

# run_test_version(args)
# test_version = [0.22756666666666667, 0.18045833333333333, 0.13184166666666666, 0.08771666666666667, 0.044908333333333335, 0.019233333333333335, 0.006183333333333333, 0.0019916666666666668, 0.00025, 5e-05]

# run_adjusted_bench(args)
#adj_bench = [2.25716667e-01, 1.80525000e-01, 1.35033333e-01, 8.60416667e-02, 4.60333333e-02, 2.01416667e-02, 6.84166667e-03, 1.50833333e-03, 4.00000000e-04, 6.66666667e-05]

