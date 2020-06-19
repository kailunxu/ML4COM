__author__ = 'yihanjiang'
'''
bcjr_rnn_train.py: train a BCJR-like RNN for Turbo Decoder.
'''
import numpy as np

from utils import corrupt_signal, snr_db2sigma, get_test_sigmas
import commpy.channelcoding.convcode as cc
from commpy.utilities import hamming_dist

import multiprocessing as mp

def benchmark_compute(args, model_decoder, PARAMS={'noise_sigma': 0, 'trellis1': 0, 'M': 0, 'tb_depth': 0}):
    
    """
    Computes model_decoder for one test_snr value
    """
    np.random.seed()

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
    num_bit_errors = hamming_dist(message_bits, decoded_bits)
    return num_bit_errors


def single_simulation_onearg(onearg):
    args, noise_sigma, trellis1, M, decoder_obj, dec_weight = onearg
    return benchmark_compute(args, decoder_obj, PARAMS = {'noise_sigma': noise_sigma, 'trellis1': trellis1, 'M': M, 'tb_depth': 15, 'dec_weight': dec_weight})


def bench_runner(args, decoder_obj, simulation_option=0, dec_weight=''):
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

        while nb_block_errors[idx] < args.num_block_err: #100
            num_block_test += args.batch_size

            if simulation_option == 0:
                # multithreading option
                onearg=(args, noise_sigma, trellis1, M, decoder_obj, dec_weight)
                oneargv=[onearg for i in range(args.batch_size)] # just repeat it batch_size times
                results1 = pool.map(single_simulation_onearg, oneargv)
            elif simulation_option == 1:
                # single simulation option
                results1 = benchmark_compute(args, decoder_obj, PARAMS = {'noise_sigma': noise_sigma, 'trellis1': trellis1, 'M': M, 'tb_depth': 15, 'dec_weight': dec_weight})
            elif simulation_option == 2:
                # run batch of simulations option
                results1 = [
                    benchmark_compute(args, decoder_obj, PARAMS = {'noise_sigma': noise_sigma, 'trellis1': trellis1, 'M': M, 'tb_depth': 15, 'dec_weight': dec_weight})
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









