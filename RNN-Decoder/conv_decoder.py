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
import commpy.channelcoding.convcode as cc

import keras.backend as K
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#frac = 0.45

#config.gpu_options.per_process_gpu_memory_fraction = frac
#set_session(tf.Session(config=config))
#print '[Test][Warining] Restrict GPU memory usage to', frac, ', enable',str(int(1.0/frac)), 'processes'
import matplotlib.pyplot as plt

import numpy as np
from conv_codes_benchmark_rewrite import conv_decode_bench


def conv_enc(X_train_raw, args):
    """
    Encodes a training sequence using cc.conv_encode(cc.Trellis([CUSTOM_GENERATOR_MATRIX])).
    """
    num_block = X_train_raw.shape[0]
    block_len = X_train_raw.shape[1]
    x_code    = []

    generator_matrix = np.array([[args.enc1, args.enc2]])
    M = np.array([args.M]) # Number of delay elements in the convolutional encoder
    trellis = cc.Trellis(M, generator_matrix,feedback=args.feedback)# Create trellis data structure

    for idx in range(num_block):
        xx = cc.conv_encode(X_train_raw[idx, :, 0], trellis)
        xx = xx[2*int(M):]
        xx = xx.reshape((block_len, 2))

        x_code.append(xx)

    return np.array(x_code)

def errors(y_true, y_pred):
    """
    Mean of a binary loss function for training and testing RNNs.
    """
    myOtherTensor = K.not_equal(y_true, K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))

def snr_db2sigma(train_snr):
    """
    Converts SNR(db units) to the variance (sigma**2) used in channel encoding.
    """
    block_len    = 100
    train_snr_Es = train_snr + 10*np.log10(float(block_len)/float(2*block_len))
    sigma_snr    = np.sqrt(1/(2*10**(float(train_snr_Es)/float(10))))
    return sigma_snr

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block', type=int, default=12000)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-variation_block_len', type=int, default=100)
    parser.add_argument('-test_ratio',  type=int, default=10)

    parser.add_argument('-num_Dec_layer',  type=int, default=2)
    parser.add_argument('-num_Dec_unit',  type=int, default=200)

    parser.add_argument('-rnn_setup', choices = ['lstm', 'gru'], default = 'gru')

    parser.add_argument('-batch_size',  type=int, default=200)
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
    parser.add_argument('-train_channel_high', type=float, default=8.0)

    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    parser.add_argument('-Dec_weight', type=str, default='default')
    parser.add_argument('-type', choices=['train', 'test', 'compareSnr', 'lengthExpand', 'bitTest', 'fig15'], default = 'compare snr')

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

    input_x         = Input(shape = (args.block_len, args.code_rate), dtype='float32', name='D_input')
    combined_x      = Lambda(channel)(input_x)

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


def bittest(args, dec_weight):
    """
    Creates a curve of Viterbi-style decoder (using this.train()) BER for individual bits using many different test-SNR rates.
    Data is encoded using this.conv_enc() and this.evaluate.channel(). Calls this.plot_bits().
    """
    X_test_raw  = np.random.randint(0,2,int(args.num_block*args.block_len/args.test_ratio))
    X_test  = X_test_raw.reshape((int(args.num_block/args.test_ratio), args.block_len, 1))
    X_conv_test  = 2.0*conv_enc(X_test, args)  - 1.0

    #print 'Testing before fine-tuning'
    snr_start = args.snr_test_start
    snr_stop  = args.snr_test_end
    snr_points = args.snr_points



    SNR_dB_start_Eb = snr_start
    SNR_dB_stop_Eb = snr_stop
    SNR_points = snr_points

    snr_interval = (SNR_dB_stop_Eb - SNR_dB_start_Eb)* 1.0 /  (SNR_points-1)
    SNRS_dB = [snr_interval* item + SNR_dB_start_Eb for item in range(SNR_points)]
    SNRS_dB_Es = [item + 10*np.log10(float(args.num_block)/float(args.num_block*2.0)) for item in SNRS_dB]
    test_sigmas = np.array([np.sqrt(1/(2*10**(float(item)/float(10)))) for item in SNRS_dB_Es])

    SNRS = SNRS_dB
    print('[testing]', SNRS_dB)

    ber = []
    for idx, snr_db in enumerate(SNRS_dB):
        print("index", idx)
        model_test = evaluate(args, dec_weight, snr_db)

        pd       = model_test.predict(X_conv_test, verbose=0)
        decoded_bits = np.round(pd)
        
        ber.append([sum(sum(abs(decoded_bits[:, i, :]-X_test[:, i, :])))*1.0/(X_test.shape[0]) for i in range(0, args.block_len)])
        
        del model_test

    print('SNRS:', SNRS_dB)
    print('BER:',ber)
    
    plot_bits(args, ber)
    
def evaluate(args, dec_weight, snr_db):
    dec_trainable = True
    inputs = Input(shape=(args.block_len, args.code_rate))

    def channel(x):
        noise_sigma =  snr_db2sigma(snr_db)
        return x+ noise_sigma*tf.random.normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)   #need to include space for different snrs

    x          = Lambda(channel)(inputs)

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

    predictions = TimeDistributed(Dense(1, activation='sigmoid'), trainable=dec_trainable, name = 'Dec_fc')(x)

    model_test = Model(inputs=inputs, outputs=predictions)

    model_test.compile(optimizer=keras.optimizers.adam(),loss=args.loss, metrics=[errors])

    model_test.load_weights(dec_weight, by_name=True)
    
    return model_test

def test(args, dec_weight, expanded = 1, bit = False):
    """
    Creates a curve of Viterbi-style decoder (using this.train()) or BCJR-style decoder (using bcjr_rnn_train.py) BER/BLER using many different test-SNR rates.
    Data is encoded using this.conv_enc() and this.evaluate.channel(). 

    Parameters
    ----------
    expanded : int
        scales the block_len of testing data by 10**expanded

    bit : bool
        If true, calls this.plot_bits() and plots the error rate for specified bit positions against the test-SNR range. Same as this.bittest().
        If false, calls this.plot_stats() and compares testing data with viterbi benchmark (using conv_codes_benchmark_rewrite.conv_codes_bench()).

    """
    if (expanded != 1):
        print("expanded the block length "+ str(expanded) +" times")
        args.block_len *= (10**expanded)
    X_test_raw  = np.random.randint(0,2,int(args.num_block*args.block_len/args.test_ratio))
    X_test  = X_test_raw.reshape((int(args.num_block/args.test_ratio), args.block_len, 1))
    X_conv_test  = 2.0*conv_enc(X_test, args)  - 1.0

    #print 'Testing before fine-tuning'
    snr_start = args.snr_test_start
    snr_stop  = args.snr_test_end
    snr_points = args.snr_points

    #dec_trainable = True

    SNR_dB_start_Eb = snr_start
    SNR_dB_stop_Eb = snr_stop
    SNR_points = snr_points

    snr_interval = (SNR_dB_stop_Eb - SNR_dB_start_Eb)* 1.0 /  (SNR_points-1)
    SNRS_dB = [snr_interval* item + SNR_dB_start_Eb for item in range(SNR_points)]
    SNRS_dB_Es = [item + 10*np.log10(float(args.num_block)/float(args.num_block*2.0)) for item in SNRS_dB]
    test_sigmas = np.array([np.sqrt(1/(2*10**(float(item)/float(10)))) for item in SNRS_dB_Es])

    SNRS = SNRS_dB
    print('[testing]', SNRS_dB)

    bler = []
    ber = []
    for idx, snr_db in enumerate(SNRS_dB):
        print("index", idx)
        model_test = evaluate(args, dec_weight, snr_db)
        pd       = model_test.predict(X_conv_test, verbose=0)
        decoded_bits = np.round(pd)
        ber_err_rate  = sum(sum(sum(abs(decoded_bits-X_test))))*1.0/(X_test.shape[0] * X_test.shape[1])# model.evaluate(X_feed_test, X_message_test, batch_size=10)
        tp0 = (abs(decoded_bits-X_test)).reshape([X_test.shape[0],X_test.shape[1]])
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
        # print ber_err_rate
        # print bler_err_rate
        if (bit == True):
            ber.append([sum(sum(abs(decoded_bits[:, i, :]-X_test[:, i, :])))*1.0/(X_test.shape[0]) for i in range(0, args.block_len)])
        else:
            ber.append(ber_err_rate)
        
        bler.append(bler_err_rate)

        del model_test

    print('[[SNRS]]:', SNRS_dB)
    print('[[BER]]:',ber)
    #print('BLER:',bler)
    # if (bit == True):
    #     plot_bits(args, ber)
    # else:
    #     if (expanded == 1):
    #         plot_stats(args, ber, args.id + " test normal ")
    #     else:
    #         plot_stats(args, ber, args.id + " length expanded " + str(expanded) + "times")
    
    return ber
    
    
def plot_compare(args, snr1, snr2, name1, name2):
    """
    Outputs a plot comparing:
        (1) viterbi benchmark (using conv_codes_benchmark_rewrite.conv_codes_bench()), 
        (2) the test BERs using train(SNR=0db), 
        (3) the test BERs using train(SNR= Uniform(0db, 8db)).
    """
    xaxis = range(int(args.snr_test_start), int(args.snr_test_end)+1)
    
    print("[[SNR1]]:", snr1)
    print("[[SNR2]]:", snr2)
    #train, = plt.plot(range(length), stats[:, 0], '-')
    ber, _ = conv_decode_bench(args)
    
    pl0, = plt.plot(xaxis, ber, '--')
    pl1, = plt.plot(xaxis, snr1, '-')
    pl2, = plt.plot(xaxis, snr2, '--')
    #bler, = plt.plot(range(length), stats[:, 2], '--')
    plot_lines = [pl0, pl1, pl2]
    plt.legend(plot_lines, ["viterbi", name1, name2])
    plt.xlabel("SNR")
    plt.ylabel("ber")
    plt.yscale("log")
    plt.grid(True, which="both", ls='--')
    plt.savefig("compare training " + args.id)
    plt.close()
    
def plot_stats(args, stats, name):
    """
    Outputs a plot comparing:
        (1) viterbi benchmark (using conv_codes_benchmark_rewrite.conv_codes_bench()) (ie. "viterbi"),
        (2) the test BERs using train(SNR=0db) (ie. "neural").
    """
    xaxis = range(int(args.snr_test_start), int(args.snr_test_end)+1)
    stats = np.array(stats)
    
    #train, = plt.plot(range(length), stats[:, 0], '-')
    
    ber, _ = conv_decode_bench(args)
    
    print("#################################")
    print("[NEURAL] ", stats)
    print("[VITERBI] ", ber)
    print("#################################")
    
    viterbi, = plt.plot(xaxis, ber, '-')
    neural, = plt.plot(xaxis, stats, '--')
    
    
    #bler, = plt.plot(range(length), stats[:, 2], '--')
    plot_lines = [neural, viterbi]
    plt.legend(plot_lines, ["neural", "viterbi"])
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
    lines = []
    legend_name = []
    test_bit = [0, 5, 20, 50, 80, 95, 99]
    
    for i in test_bit:
        print(stats[:, i])
        line, = plt.plot(xaxis, stats[:, i], '-')
        lines.append(line)
        legend_name.append("bit position = " + str(i))
    
    plt.legend(lines, legend_name)
    plt.xlabel("SNR")
    plt.ylabel("ber")
    plt.yscale("log")
    plt.grid(True, which="both", ls='--')
    plt.savefig("bit" + args.id)
    
    plt.close()



if __name__ == '__main__':
    
    args = get_args()
    if (args.type == 'train'):
        #python3 conv_decoder.py -type train -train_channel_low 0 -train_channel_high 0
        print("Training:")
        train(args)
    if (args.type == 'test'): # 382020
        #python3 conv_decoder.py -type test
        print("Testing:")
        #modelnum = input("Please input the model number: ")
        #modelnum = "382020"
        modelnum = args.id
        test(args, dec_weight='./tmp/conv_dec'+modelnum+'.h5')
    if (args.type == 'compareSnr'):
        print("Compare two snr training result:")
        model1 = input("please input the first model number to be compared: ")
        model2 = input("please input the second model number to be compared: ")
        #snr1 = test(args, dec_weight='./tmp/conv_dec'+model1+'.h5')
        snr1 = [0.23293333333333333, 0.192025, 0.14896666666666666, 0.10428333333333334, 0.06496666666666667, 0.032966666666666665, 0.014891666666666666, 0.006516666666666666, 0.0020583333333333335, 0.0007916666666666666]
        #snr2 = test(args, dec_weight='./tmp/conv_dec'+model2+'.h5')
        snr2 = [0.2396875, 0.1890625, 0.1359375, 0.09375, 0.0571875, 0.0278125, 0.006853932584269663, 0.0017366946778711485, 0.00025, 0.0001]
        plot_compare(args, snr1, snr2, "train snr=0db", "train snr=0-8db")
    if (args.type == 'lengthExpand'):
        print("Expand length Test:")
        modelnum = input("Please input the model number: ")
        ratio = int(input("please input the ratio to expand: "))
        test(args, dec_weight='./tmp/conv_dec'+modelnum+'.h5', expanded = ratio)
    if (args.type == 'bitTest'):
        #python3 conv_decoder.py -type 'bit test'
        print("Bit Test:")
        modelnum = input("Please input the model number: ")
        test(args, dec_weight='./tmp/conv_dec'+modelnum+'.h5', bit = True)
    if (args.type == 'fig15'):
        print("Running Fig 15 Test:")
        SNR_train_range = [-1, 1, 2] # zero excluded due to type="test"
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
        #-2: 846991
        #-1: 
        # 1: 
        # 2: 
        # RUN plot compare on Log scale for all of them
        # To find minimum, run:
        ##############################
        # min_val = {i:1000 for i in range(-3, 6)}
        # min_id = {i:1000 for i in range(-3, 6)}
        # train_ids = [-2.5, -2, -1.5, -1, -0.5, 0, .5, 1, 1.5, 2]
        # for i in range(len(train_ids)):
        #    for j in train_ids:
        #      if V[j][i] < min_val[i-3]
        #        min_val[i-3] = V[j][i]
        #        min_id[i-3] = j
        # V = {0: ber = [0.2346, 0.1893, 0.14448333333333332, 0.095325, 0.0534, 0.025833333333333333, 0.010116666666666666, 0.0031333333333333335, 0.0009]}
        ##############################



        