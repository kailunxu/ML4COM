# ML4COM
This is the github repository for the summer project.

Our first subproject this summer consists of reproducing the results of [1]. We aim to become comfortable with communication concepts and their corresponding machine learning parts in [2]. In short, [1] sought to train a bi-GRU-based RNN to decode RSC codes over AWGN channels. The authors varied the code rate and signal-to-noise ratios of their deterministic encoders and used random binary data streams as known in Monte Carlo simulations. They then compared the bit (block) error rate BER (BLER) of their NN-decoder with benchmarks using the Viterbi (BCJR, MAP) algorithm.

[1] H. Kim, Y. Jiang, R. Rana, S. Kannan, S. Oh, and P. Viswanath, “Communication algorithms via deep learning,” in The International Conference on Representation Learning (ICLR 2018) Proceedings. Vancouver, 2018.
[2] https://github.com/yihanjiang/Sequential-RNN-Decoder