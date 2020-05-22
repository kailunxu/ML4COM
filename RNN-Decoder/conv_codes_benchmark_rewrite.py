'''
Evaluate convolutional code benchmark.
'''
#from utils import corrupt_signal, get_test_sigmas

import sys
import numpy as np

import commpy.channelcoding.convcode as cc
from commpy.utilities import hamming_dist
import multiprocessing as mp
from utils import corrupted_signal

def get_test_sigmas(snr_start, snr_end, snr_points):
    SNR_dB_start_Eb = snr_start
    SNR_dB_stop_Eb = snr_end
    SNR_points = snr_points

    snr_interval = (SNR_dB_stop_Eb - SNR_dB_start_Eb)* 1.0 /  (SNR_points-1)
    SNRS_dB = [snr_interval* item + SNR_dB_start_Eb for item in range(SNR_points)]
    SNRS_dB_Es = [item + 10*np.log10(1.0/2.0) for item in SNRS_dB]
    test_sigmas = np.array([np.sqrt(1/(2*10**(float(item)/float(10)))) for item in SNRS_dB_Es])

    SNRS = SNRS_dB
    print('[testing] SNR range in dB ', SNRS)

    return SNRS, test_sigmas


def turbo_compute(args, idx, x, trellis1, test_sigmas, M):
    '''
    Compute Turbo Decoding in 1 iterations for one SNR point.
    '''
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

def conv_decode_bench(args):
    
    num_block = 100
    ##########################################
    # Setting Up Codec
    ##########################################
    M = np.array([2]) # Number of delay elements in the convolutional encoder
    generator_matrix = np.array([[args.enc1, args.enc2]])
    feedback = args.feedback

    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)  # Create trellis data structure

    SNRS, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)

    tb_depth = 15

    commpy_res_ber = []
    commpy_res_bler= []

    nb_errors          = np.zeros(test_sigmas.shape)
    map_nb_errors      = np.zeros(test_sigmas.shape)
    nb_block_no_errors = np.zeros(test_sigmas.shape)

    for idx in range(len(test_sigmas)):
        results = []
        print(num_block)
        #pool = mp.Pool(processes=args.num_cpu)
        #results = pool.starmap(turbo_compute, [(idx,x) for x in range(num_block)])
        for x in range(num_block):
            results.append(turbo_compute(args, idx, x, trellis1, test_sigmas, M))
        for result in results:
            if result == 0:
                nb_block_no_errors[idx] = nb_block_no_errors[idx]+1
                
        nb_errors[idx]+= sum(results)
        #print('[testing]SNR: ' , SNRS[idx])
        print('[testing]BER: ', sum(results)/float(args.block_len*num_block))
        #print('[testing]BLER: ', 1.0 - nb_block_no_errors[idx]/args.num_block)
        commpy_res_ber.append(sum(results)/float(args.block_len*num_block))
        commpy_res_bler.append(1.0 - nb_block_no_errors[idx]/num_block)


    print('[Result]SNR: ', SNRS)
    print('[Result]BER', commpy_res_ber)
    print('[Result]BLER', commpy_res_bler)


    return commpy_res_ber, commpy_res_bler