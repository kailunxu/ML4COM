'''
Evaluate convolutional code benchmark.
'''

import sys
import numpy as np

from utils import corrupt_signal, get_test_sigmas
from run_benchmarks import benchmark_compute, bench_runner
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


def decoder_obj(args, received, PARAMS={'noise_sigma': 0, 'trellis1': 0, 'M': 0, 'tb_depth': 0}):
    return cc.viterbi_decode(received.astype(float), PARAMS['trellis1'], PARAMS['tb_depth'], decoding_type='unquantized')

def conv_decode_bench_without_threads(args):
    """
    DEPRECATED.
    Outputs benchmark for viterbi algorithm for a given range of test_snr values.
    """
    print("viterbi starts (block_len =", args.block_len, ")")

    # Setting Up Codec
    M = np.array([2]) # Number of delay elements in the convolutional encoder
    generator_matrix = np.array([[args.enc1, args.enc2]])
    feedback = args.feedback
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)  # Create trellis data structure

    SNRS, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)

    commpy_res_ber = []

    nb_errors          = np.zeros(test_sigmas.shape)
    nb_block_no_errors = np.zeros(test_sigmas.shape)

    for idx in range(len(test_sigmas)):
        print('[testing]SNR: %4.1f'% SNRS[idx])

        num_block_test = args.num_block
        results = []
        for x in range(args.num_block):
            val = benchmark_compute(args, decoder_obj, PARAMS = {'noise_sigma': test_sigmas[idx],
                                                                 'trellis1': trellis1, 
                                                                 'M': M,
                                                                 'tb_depth': 15})
            results.append(val)
            # if (sum(results)>60 and x > num_block /40):
            num_block_test = x + 1
            # break
        nb_block_no_errors[idx] += sum(np.array(results)==0)

        nb_errors[idx]+= sum(results)
        BER = sum(results)/float(args.block_len*num_block_test)
    
        commpy_res_ber.append(BER)
    
    print('[Result]SNR: ', SNRS)
    print('[Result]BER', commpy_res_ber)

    return commpy_res_ber, 0#, commpy_res_bler


if __name__ == '__main__':
    args = get_args()
    # conv_decode_bench(args)
    bench_runner(args, decoder_obj)
