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
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.backend.tensorflow_backend import set_session

import numpy as np
import matplotlib.pyplot as plt

import commpy.channelcoding.convcode as cc
from commpy.utilities import hamming_dist

# from conv_codes_benchmark_rewrite_aa import conv_decode_bench
# from bcjr_rnn_train import bcjr_bench
from utils import get_test_sigmas, errors, snr_db2sigma, conv_enc
from run_benchmarks import bench_runner



def get_test_model(args, dec_weight, noise_sigma):
    """
    Returns
    -------
    model_test
        Returns a model with trained weights according to [dec_weight] and the given [snr_db]
    """
    dec_trainable = True
    inputs = Input(shape=(args.block_len, args.code_rate))

    def channel(x):
        """
        Simulates an AWGN noisy channel, outputting x+noise_sigma*Normal(0,1).
        """
        # noise_sigma =  snr_db2sigma(snr_db)
        return x+ noise_sigma*tf.random.normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)   #need to include space for different snrs

    # Simulate channel
    x          = Lambda(channel)(inputs)

    # Establish RNN tensorflow model
    for layer in range(args.num_Dec_layer - 1):
        if args.rnn_setup == 'lstm':
            x = Bidirectional(LSTM(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                 trainable=dec_trainable), name = 'Dec_'+args.rnn_setup+'_'+str(layer))(x)
        elif args.rnn_setup == 'gru':
            x = Bidirectional(GRU(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                                 trainable=dec_trainable), name = 'Dec_'+args.rnn_setup+'_'+str(layer))(x)

        x = BatchNormalization(trainable=dec_trainable, name = 'Dec_bn_'+str(layer))(x)

    y = x

    if args.rnn_setup == 'lstm':
        y = Bidirectional(LSTM(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                            trainable=dec_trainable), name = 'Dec_'+args.rnn_setup+'_'+str(args.num_Dec_layer-1) )(y)
    elif args.rnn_setup == 'gru':
        y = Bidirectional(GRU(units=args.num_Dec_unit, activation='tanh', return_sequences=True,
                            trainable=dec_trainable), name = 'Dec_'+args.rnn_setup+'_'+str(args.num_Dec_layer-1) )(y)

    x = BatchNormalization(trainable=dec_trainable, name = 'Dec_bn_'+str(args.num_Dec_layer-1))(y)

    # Add sigmoid layer
    predictions = TimeDistributed(Dense(1, activation='sigmoid'), trainable=dec_trainable, name = 'Dec_fc')(x)

    # Compile and load model
    model_test = Model(inputs=inputs, outputs=predictions)
    model_test.compile(optimizer=keras.optimizers.adam(),loss=args.loss, metrics=[errors])
    model_test.load_weights(dec_weight, by_name=True)
    
    return model_test

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

def build_decoder(args):
    """
    Builds the outline of a Keras-based RNN. 
    [[ Input -> AWGN channel -> (Bi-GRU + BatchNorm)*2 -> Sigmoid. ]]
    Uses dropout_rate=1, tanh activation, and Adam optimization.
    """

    ont_pretrain_trainable = True
    dropout_rate           = 1.0

    def channel(x):
        """
        Simulates an AWGN noisy channel, outputting x+Uniform(_,_)*Normal(0,1).
        """
        print('training with noise snr db', args.train_channel_low, args.train_channel_high)
        noise_sigma_low =  snr_db2sigma(args.train_channel_low) # 0dB
        noise_sigma_high =  snr_db2sigma(args.train_channel_high) # 0dB
        print('training with noise snr db', noise_sigma_low, noise_sigma_high)
        noise_sigma =  tf.random.uniform(tf.shape(x),
            minval=noise_sigma_high,
            maxval=noise_sigma_low,
            dtype=tf.float32
        )

        return x+ noise_sigma*tf.random.normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)   #need to include space for different snrs

    # Simulate channel
    input_x         = Input(shape = (args.block_len, args.code_rate), dtype='float32', name='D_input')
    combined_x      = Lambda(channel)(input_x)

    # Establish RNN tensorflow model
    for layer in range(args.num_Dec_layer):
        if args.rnn_setup == 'gru':
            combined_x = Bidirectional(GRU(units=args.num_Dec_unit, activation='tanh', dropout=dropout_rate,
                                           return_sequences=True, trainable=ont_pretrain_trainable),
                                       name = 'Dec_'+args.rnn_setup+'_'+str(layer))(combined_x)
        else:
            combined_x = Bidirectional(LSTM(units=args.num_Dec_unit, activation='tanh', dropout=dropout_rate,
                                            return_sequences=True, trainable=ont_pretrain_trainable),
                                       name = 'Dec_'+args.rnn_setup+'_'+str(layer))(combined_x)

        combined_x = BatchNormalization(name = 'Dec_bn'+'_'+str(layer), trainable=ont_pretrain_trainable)(combined_x)

    # Add sigmoid layer
    decode = TimeDistributed(Dense(1, activation='sigmoid'), trainable=ont_pretrain_trainable, name = 'Dec_fc')(combined_x)  #sigmoid

    return Model(input_x, decode)

def train(args):
    """
    Trains a Viterbi-style decoder using two bi-GRU + batch norm layers followed by sigmoid.
    Data is encoding using this.conv_enc() and this.build_decoder.channel().
    Outputs a .h5 file with saved RNN weights.
    """

    X_train_raw = np.random.randint(0,2,args.block_len * args.num_block)
    X_test_raw  = np.random.randint(0,2,int(args.block_len * args.num_block/args.test_ratio))

    X_train = X_train_raw.reshape((args.num_block, args.block_len, 1))
    X_test  = X_test_raw.reshape((int(args.num_block/args.test_ratio), args.block_len, 1))

    X_conv_train = 2.0*conv_enc(X_train, args) - 1.0
    X_conv_test  = 2.0*conv_enc(X_test, args)  - 1.0

    model = build_decoder(args)

    def scheduler(epoch):
        """
        Learning rate scheduler for RNN training
        """
        if epoch > 10 and epoch <=15:
            print('changing by /10 lr')
            lr = args.learning_rate/10.0
        elif epoch >15 and epoch <=20:
            print('changing by /100 lr')
            lr = args.learning_rate/100.0
        elif epoch >20 and epoch <=25:
            print('changing by /1000 lr')
            lr = args.learning_rate/1000.0
        elif epoch > 25:
            print('changing by /10000 lr')
            lr = args.learning_rate/10000.0
        else:
            lr = args.learning_rate

        return lr
    change_lr = LearningRateScheduler(scheduler)


    if args.Dec_weight == 'default':
        print('Decoder has no weight')
    else:
        print('Decoder loaded weight', args.Dec_weight)
        model.load_weights(args.Dec_weight)

    print("learning rate", args.learning_rate)
    optimizer = Adam(args.learning_rate)

    # Build and compile the discriminator
    model.compile(loss=args.loss,  optimizer=optimizer, metrics=[errors])
    model.summary()

    model.fit(X_conv_train,X_train, validation_data=(X_conv_test, X_test),
              callbacks = [change_lr],
              batch_size=args.batch_size, epochs=args.num_epoch)

    model.save_weights('./tmp/conv_dec'+args.id+'.h5')

def decoder_obj(args, received, PARAMS={'dec_weight': 0, 'noise_sigma': 0, 'M': 0, 'model_test': 0}):
    # model_test = get_test_model(args, PARAMS['dec_weight'], PARAMS['noise_sigma'])
    pd         = PARAMS['model_test'].predict(received, verbose=0)
    decoded_bits = np.round(pd).astype(int)
    return decoded_bits


def plot_compare(args, snr1, snr2, name1, name2):
    """
    Outputs a plot comparing:
        (1) viterbi benchmark (using conv_codes_benchmark_rewrite.conv_codes_bench()), 
        (2) the test BERs using train(SNR=0db), 
        (3) the test BERs using train(SNR= Uniform(0db, 8db)).
    """
    print("[[SNR1]]:", snr1)
    print("[[SNR2]]:", snr2)

    xaxis = range(int(args.snr_test_start), int(args.snr_test_end)+1)
    viterb, _ = conv_decode_bench(args)
    
    #bler, = plt.plot(range(length), stats[:, 2], '--')
    plt.plot(xaxis, viterb, '.--')
    plt.plot(xaxis, snr1, '.-')
    plt.plot(xaxis, snr2, '.--')

    plt.legend(["viterbi", name1, name2])
    plt.xlabel("SNR")
    plt.ylabel("ber")
    plt.yscale("log")
    plt.grid(True, which="both", ls='--')
    plt.savefig("compare training " + args.id)
    plt.close()
    
def plot_stats(args, stats, name):
    """
    Outputs a plot comparing:
        (1) viterbi benchmark (using conv_codes_benchmark_rewrite.conv_codes_bench()),
        (2) map benchmark (using bcjr_rnn_train.bcjr_bench()),
        (3) the test BERs using train(SNR=0db) (ie. "neural").
    """
    xaxis = range(int(args.snr_test_start), int(args.snr_test_end)+1)
    stats = np.array(stats)
    # viterbi_bench, _ = conv_decode_bench(args)
    # map_bench, _ = bcjr_bench(args)
    viterbi_bench = [0.24605960264900661, 0.19665562913907284, 0.1459933774834437, 0.0870860927152318, 0.04794701986754967, 0.01652317880794702, 0.004933774834437086, 0.0013606911447084232, 0.0002955426356589147, 6.646655231560892e-05]
    map_bench = [0.21665, 0.17735, 0.13275833333333334, 0.08685, 0.0436, 0.017833333333333333, 0.005308333333333333, 0.0010416666666666667, 0.00023333333333333333, 4.1666666666666665e-05]

    plt.plot(xaxis, viterbi_bench, '.-')
    plt.plot(xaxis, map_bench, '.-')
    plt.plot(xaxis, stats, '.--')
    
    plt.legend(["viterbi", "map", "neural"])
    plt.xlabel("SNR")
    plt.ylabel("ber")
    plt.yscale("log")
    plt.grid(True, which="both", ls='--')
    plt.savefig(name)
    plt.close()

def plot_bits(args, stats):
    """
    Called using this.test(bit = True). 
    Outputs a plot comparing each of the [0, 5, 20, ..., 99]th bit error rates against a test-SNR range.
    """
    xaxis = range(int(args.snr_test_start), int(args.snr_test_end)+1)
    stats = np.array(stats)
    legend_name = []
    test_bit = [0, 5, 20, 50, 80, 95, 99]
    
    for i in test_bit:
        #print(stats[:, i])
        plt.plot(xaxis, stats[:, i], '.-')
        legend_name.append("bit position = " + str(i))
    
    plt.legend(legend_name)
    plt.xlabel("SNR")
    plt.ylabel("ber")
    plt.yscale("log")
    plt.grid(True, which="both", ls='--')
    plt.savefig("bit" + args.id)
    
    plt.close()

def plot_fig15(args, V):
    """
    Plots figure 15 given data.
    """
    min_val = {i:1000 for i in range(-3, 6)}
    min_id = {i:1000 for i in range(-3, 6)}
    train_ids = [-2.5, -2, -1.5, -1, -0.5, 0, .5, 1, 1.5, 2]

    for i in range(len(train_ids))-1:
        for j in train_ids:
            if V[j][i] < min_val[i-3]:
                min_val[i-3] = V[j][i]
                min_id[i-3] = j

    
    # Plot train_SNR vs test_SNR
    legend_names = []
    for k,v in V:
        plt.plot(range(-3,6), v, ".-")
        legend_names.append("train SNR = " + str(k))

    plt.legend(legend_names)
    plt.xlabel("SNR")
    plt.ylabel("ber")
    plt.yscale("log")
    plt.grid(True, which="both", ls='--')
    plt.savefig("Fig15A_" + str(args.id))
    plt.close()
    
    # Plot minimum
    plt.clear()
    plt.plot(range(-3,6), min_id.values(), '.-')
    plt.xlabel("test SNR")
    plt.ylabel("best train SNR")
    plt.legend("SNR rate 1/2")
    plt.savefig("Fig15B_" + str(args.id))
    plt.close()
    

def main(type):
    """
    Parameters
    ----------
    type : str
        'train': 
            $ python3 conv_decoder.py -type train -train_channel_low 0 -train_channel_high 0
            Trains a model with user provided arguments. Calls this.train().
        'test':
            $ python3 conv_decoder.py -type test
            Tests a pretrained model. Calls this.test().
        'test_batches':
            Tests in batches with stopping condition. Multithreading not yet implemented as of 6/18/20
        'compareSNR':
            Outputs a plot comparing the BER rates of two provided pretrained models. Calls this.test() and this.plot_compare().
        'lengthExpand':
            Tests a pretrained model, but with block length scaled by a certain factor (10**expanded). Calls this.test(). See Fig 4.
        'bitTest':
            $ python3 conv_decoder.py -type 'bitTest'
            Tests a pretrained model, but outputs plot with bit errors of certain positions in codewords, rather than the BER themselves.
        'fig15':
            Reproduces Figure 15 by running this.train() nine times over different train_SNR parameters. 
    """

    if (args.type == 'train'):
        print("Training:")
        train(args)
    if (args.type == 'test_batches'):
        print("Testing in Batches:")
        modelnum = input("Please input the model number: ")
        simulation_option = int(input("Please input simulation option {0,1,2}: "))
        # modelnum = args.id
        # test_in_batches(args, dec_weight='./tmp/conv_dec'+modelnum+'.h5')
        bench_runner(args, decoder_obj, simulation_option=simulation_option, dec_weight='./tmp/conv_dec'+modelnum+'.h5', get_test_model=get_test_model)
    if (args.type == 'test'):
        print("(DEPCREATED) Testing:")
        modelnum = input("Please input the model number: ")
        #modelnum = args.id
        test(args, dec_weight='./tmp/conv_dec'+modelnum+'.h5')
    if (args.type == 'compareSnr'):
        print("Compare two snr training result:")
        model1 = input("please input the first model number to be compared: ")
        model2 = input("please input the second model number to be compared: ")
        snr1 = test(args, dec_weight='./tmp/conv_dec'+model1+'.h5')
        # snr1 = [0.23293333333333333, 0.192025, 0.14896666666666666, 0.10428333333333334, 0.06496666666666667, 0.032966666666666665, 0.014891666666666666, 0.006516666666666666, 0.0020583333333333335, 0.0007916666666666666]
        snr2 = test(args, dec_weight='./tmp/conv_dec'+model2+'.h5')
        # snr2 = [0.2396875, 0.1890625, 0.1359375, 0.09375, 0.0571875, 0.0278125, 0.006853932584269663, 0.0017366946778711485, 0.00025, 0.0001]
        plot_compare(args, snr1, snr2, "train snr=0db", "train snr=0-8db")
    if (args.type == 'lengthExpand'):
        print("Expand length Test:")
        modelnum = input("Please input the model number: ")
        expanded = int(input("please input the ratio to expand: "))
        test(args, dec_weight='./tmp/conv_dec'+modelnum+'.h5', expanded = expanded)
    if (args.type == 'bitTest'):
        print("Bit Test:")
        modelnum = input("Please input the model number: ")
        test(args, dec_weight='./tmp/conv_dec'+modelnum+'.h5', bit = True)
    if (args.type == 'fig15'):
        print("Running Fig 15 Test:")
        SNR_train_range = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        V = {}
        args.snr_test_start = -3
        args.snr_test_end = 5
        args.snr_points = 9
        for y in SNR_train_range:
            args.train_channel_low = y
            args.train_channel_high = y
            args.id = str(np.random.random())[2:8]
            print("=================================")
            print("RUNNING Y=", y, "; ID = ", args.id)
            print("=================================")
            train(args)
            V[y] = test(args, dec_weight='./tmp/conv_dec'+args.id+'.h5')
            print("=================================")
            print("=================================")
            print("V so far...\n", V)
            print("=================================")
        print("\n\n\nDONE\n")
        print(V)
        plot_fig15(args, V)


###########################################################

def test(args, dec_weight, expanded = 1, bit = False):
    """
    DEPRECATED

    Creates a curve of Viterbi-style decoder (using this.train()) or BCJR-style decoder (using bcjr_rnn_train.py) BER/BLER using many different test-SNR rates.
    Data is encoded using this.conv_enc() and this.get_test_model.channel(). 

    Parameters
    ----------
    expanded : int
        scales the block_len of testing data by 10**expanded. See Figure 4

    bit : bool
        If true, calls this.plot_bits() and plots BER for individually specified bit positions against the test-SNR range.
        If false, calls this.plot_stats() and compares testing data with viterbi benchmark (using conv_codes_benchmark_rewrite.conv_codes_bench()).

    """
    # Expand block length if desired
    if (expanded != 1):
        print("expanded the block length "+ str(expanded) +" times")
        args.block_len *= (10**expanded)

    # Get testing data
    X_test_raw  = np.random.randint(0,2,int(args.num_block*args.block_len/args.test_ratio))
    X_test  = X_test_raw.reshape((int(args.num_block/args.test_ratio), args.block_len, 1))
    X_conv_test  = 2.0*conv_enc(X_test, args)  - 1.0

    SNRS_dB, _ = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)
    print('[testing]', SNRS_dB)

    bler = []
    ber = []
    for idx, snr_db in enumerate(SNRS_dB):
        print("index", idx)
        # get model predictions
        model_test = get_test_model(args, dec_weight, snr_db)
        pd       = model_test.predict(X_conv_test, verbose=0)
        decoded_bits = np.round(pd)

        # ber error rate
        if bit == True:
            ber_err_rate = [sum(sum(abs(decoded_bits[:, i, :]-X_test[:, i, :])))*1.0/(X_test.shape[0]) for i in range(0, args.block_len)]
        else:
            ber_err_rate  = sum(sum(sum(abs(decoded_bits-X_test))))*1.0/(X_test.shape[0] * X_test.shape[1])
            # model.get_test_model(X_feed_test, X_message_test, batch_size=10)
        ber.append(ber_err_rate)

        # bler error rate
        tp0 = (abs(decoded_bits-X_test)).reshape([X_test.shape[0],X_test.shape[1]])
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
        bler.append(bler_err_rate)

        del model_test

    print('[[SNRS]]:', SNRS_dB)
    print('[[BER]]:',ber)
    # print('BLER:',bler)
    if (bit == True):
        plot_bits(args, ber)
    else:
        plot_name = args.id + (" test normal " if expanded ==1 else " length expanded " + str(expanded) + "times")
        plot_stats(args, ber, plot_name)

    return ber

def single_simulation(args, dec_weight, snr_db, num_blocks):
    """
    DEPRECATED
    """
    X_test  = np.random.randint(0,2,(num_blocks,args.block_len,1))
    X_conv_test  = 2.0*conv_enc(X_test, args)  - 1.0

    # get model predictions
    model_test   = get_test_model(args, dec_weight, snr_db)
    pd           = model_test.predict(X_conv_test, verbose=0)
    decoded_bits = np.round(pd)

    # num_bit_errors = hamming_dist(X_test.flatten(), decoded_bits.flatten())
    num_bit_errors = sum(X_test.flatten() != decoded_bits.flatten())
    
    del model_test
    return num_bit_errors
    

def test_in_batches(args, dec_weight):
    """
    DEPRECATED

    Creates a curve of Viterbi-style decoder (using this.train()) or BCJR-style decoder (using bcjr_rnn_train.py) BER/BLER using many different test-SNR rates.
    Data is encoded using this.conv_enc() and this.get_test_model.channel(). 
    """
    print("running TIB")

    # Get testing data
    num_blocks = int(args.num_block/args.test_ratio)

    SNRS, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)
    
    commpy_res_ber = []
    commpy_res_bler= []

    nb_errors          = np.zeros(test_sigmas.shape)
    nb_block_errors = np.zeros(test_sigmas.shape)

    for idx, snr_db in enumerate(SNRS):
        print('[testing]SNR: %4.1f'% SNRS[idx])

        num_block_test = 0
        
        while nb_block_errors[idx] < args.num_block_err: # 100
            print("Batch: ", num_block_test)
            num_block_test += args.batch_size # run a batch of batch_size simulations

            results1 = [single_simulation(args, dec_weight, snr_db, num_blocks) for i in range(args.batch_size)]
            # results1 = [
            #     benchmark_compute(args, decoder_obj, {'dec_weight': dec_weight, 'snr_db': snr_db, 'num_blocks': num_blocks},isConvDecode=True)
            #     for i in range(args.batch_size)
            # ]
            
            nb_block_errors[idx] += sum(np.array(results1) > 0)
            nb_errors[idx] += sum(results1)

            BER = nb_errors[idx]/float(args.block_len*num_block_test)
            BLER = nb_block_errors[idx]/float(num_block_test)

            if num_block_test % 100 ==0:
                print('%8d %8d %8d %8.2e %8.2e'% (num_block_test, int(nb_block_errors[idx]), nb_errors[idx] ,BLER,BER))

        print('%8d %8d %8d %8.2e %8.2e'% (num_block_test, int(nb_block_errors[idx]), nb_errors[idx] ,BLER,BER))
        commpy_res_ber.append(BER)
        commpy_res_bler.append(BLER)


        print('[testing]BLER: %8.2e'% BLER)
        print('[testing]BER:  %8.2e'% BER)

    print('[[SNRS]]:', SNRS)
    print('[[BER]]:', commpy_res_ber)
    # print('[[BLER]]:',commpy_res_bler)
    # plot_stats(args, commpy_res_ber, "test with batches")

    return commpy_res_ber
      

if __name__ == '__main__':
    args = get_args()
    main(args.type)

        