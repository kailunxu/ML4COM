'''
Evaluate convolutional code benchmark.
'''

import sys
import numpy as np

from utils_aa import corrupt_signal, get_test_sigmas
import commpy.channelcoding.convcode as cc
from commpy.utilities import hamming_dist
import multiprocessing as mp


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block_err', type=int, default=100)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-num_cpu', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=100)

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
    
    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    args = parser.parse_args()
    print(args)

    print('[ID]', args.id)
    return args

def single_simulation_onearg(onearg):
    args=onearg[0]
    sigma_val=onearg[1]
    trellis1=onearg[2]
    M=onearg[3]
    return single_simulation(args, sigma_val, trellis1, M)

def single_simulation(args, sigma_val, trellis1, M):
    '''
    Helper function used to compute Viterbi decoding for one SNR point and one codeword. 
    Called in this.conv_decode_bench().
    '''
    tb_depth = 15
    np.random.seed()
    message_bits = np.random.randint(0, 2, args.block_len)

    coded_bits = cc.conv_encode(message_bits, trellis1)
    received  = corrupt_signal(coded_bits, noise_type =args.noise_type, sigma = sigma_val,
                               vv =args.v, radar_power = args.radar_power, radar_prob = args.radar_prob,
                               denoise_thd = args.radar_denoise_thd)

    # make fair comparison between (100, 204) convolutional code and (100,200) RNN decoder, set the additional bit to 0
    received[-2*int(M):] = 0.0

    decoded_bits = cc.viterbi_decode(received.astype(float), trellis1, tb_depth, decoding_type='unquantized')
    decoded_bits = decoded_bits[:-int(M)]
    num_bit_errors = hamming_dist(message_bits, decoded_bits)
    return num_bit_errors

def conv_decode_bench(args):
    """
    Outputs benchmark for viterbi algorithm for a given range of test_snr values. 
    Called in plot_stats() of conv_decoder.py.
    """
    print("viterbi starts (block_len =", args.block_len, ")")

    # Setting Up Codec
    M = np.array([2]) # Number of delay elements in the convolutional encoder
    generator_matrix = np.array([[args.enc1, args.enc2]])
    feedback = args.feedback
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)  # Create trellis data structure

    SNRS, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)

    commpy_res_ber = []
    commpy_res_bler= []

    nb_errors          = np.zeros(test_sigmas.shape)
    nb_block_errors = np.zeros(test_sigmas.shape)

    for idx in range(len(test_sigmas)):
        print('[testing]SNR: %4.1f'% SNRS[idx])
        
        num_block_test = 0
        pool = mp.Pool(processes=args.num_cpu)
        
        while nb_block_errors[idx] < args.num_block_err: # 100
            num_block_test += args.batch_size # run a batch of battch_size simulations
            
            # multithreading option
            onearg=(args, test_sigmas[idx], trellis1, M)
            oneargv=[onearg for i in range(args.batch_size)] # just repeat it batch_size times
            results1 = pool.map(single_simulation_onearg, oneargv)
            
            # single simulation option
            # results1=single_simulation(args, test_sigmas[idx], trellis1, M)

            # run batch of sumulations option
            # results1=[single_simulation_onearg(onearg) for i in range(args.batch_size)]
            
            nb_block_errors[idx] += sum(np.array(results1) > 0)
            nb_errors[idx] += sum(results1)
            
            BER = nb_errors[idx]/float(args.block_len*num_block_test)
            BLER = nb_block_errors[idx]/float(num_block_test)

            if num_block_test % 100 ==0: # print intermeadiate results every so often
                print('%8d %8d %8d %8.2e %8.2e'% (num_block_test, int(nb_block_errors[idx]), nb_errors[idx] ,BLER,BER))

        
        pool.close()
        print('%8d %8d %8d %8.2e %8.2e'% (num_block_test, int(nb_block_errors[idx]), nb_errors[idx] ,BLER,BER))

        print('[testing]BLER: %8.2e'% BLER)
        print('[testing]BER:  %8.2e'% BER)
        print('')
        commpy_res_ber.append(BER)
        commpy_res_bler.append(BLER)


    print('[Result]SNR:  '),
    for i in SNRS:
        print('%8.1f'% i),
    print('')
    print('[Result]BLER: '),
    for i in commpy_res_bler:
        print('%8.2e'% i),
    print('')
    print('[Result]BER:  '),
    for i in commpy_res_ber:
        print('%8.2e'% i),

    return commpy_res_ber, 0#, commpy_res_bler

if __name__ == '__main__':
    args = get_args()
    conv_decode_bench(args)
