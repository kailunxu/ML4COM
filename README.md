# ML4COM
This [3] is a github repository for the ML4COM summer project, May-Aug 2020. Authors include Professor Achilleas Anastasopoulos, Joshua Kavner, and Kailun Xu.

Our first subproject this summer consists of reproducing the results of [1]. We aim to become comfortable with communication concepts and their corresponding machine learning parts in [2]. In summary, [1] sought to train a bi-GRU-based RNN to decode RSC codes over AWGN channels. The authors varied the code rate and signal-to-noise ratios of their deterministic encoders and used random binary data streams as known in Monte Carlo simulations. They then compared the bit (block) error rate BER (BLER) of their NN-decoder with benchmarks using the Viterbi (BCJR, MAP) algorithm.

###
Run the following:

python conv_codes_benchmark_rewrite_aa.py -num_block_err 100 -block_len 100 -num_cpu=4 -snr_test_start 4.0 -snr_test_end 4.0 -snr_points 1

compare with the parameter: -num_cpu=1 and -num_cpu=2

Note that there is a new argument "num_block_err" which is the number of block errors to collect for a simulation.
###

[1] H. Kim, Y. Jiang, R. Rana, S. Kannan, S. Oh, and P. Viswanath, “Communication algorithms via deep learning,” in The International Conference on Representation Learning (ICLR 2018) Proceedings. Vancouver, 2018.
[2] https://github.com/yihanjiang/Sequential-RNN-Decoder
[3] https://github.com/kailunxu/ML4COM
