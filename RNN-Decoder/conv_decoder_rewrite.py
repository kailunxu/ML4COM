import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model_GRU import RNNGRU
import sys
import numpy as np
import commpy.channelcoding.convcode as cc
import matplotlib.pyplot as plt
from conv_codes_benchmark_rewrite import conv_decode_bench

def conv_enc(X_train_raw, args):
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

def _generate_training_data(args):
    '''
    returns the training data
    '''
    X_train_raw = np.random.randint(0,2,args.block_len * args.num_block)
    Y_train = X_train_raw.reshape((args.num_block, args.block_len, 1))
    X_conv_train = 2.0*conv_enc(Y_train, args) - 1.0
    Y_train = torch.tensor(Y_train.reshape((int(args.num_block / args.batch_size), args.batch_size, args.block_len, 1)))
    X_conv_train = torch.tensor(X_conv_train.reshape((int(args.num_block / args.batch_size), args.batch_size, args.block_len, 2)))
    print(X_conv_train.shape)
    print(Y_train.shape)
    return X_conv_train.float(), Y_train.float()

def _generate_test_data(args):
    '''
    returns the training data
    '''
    X_train_raw = np.random.randint(0,2,args.block_len * args.test_block)
    Y_train = X_train_raw.reshape((args.test_block, args.block_len, 1))
    X_conv_train = 2.0*conv_enc(Y_train, args) - 1.0
    Y_train = torch.tensor(Y_train.reshape((int(args.test_block / args.batch_size), args.batch_size, args.block_len, 1)))
    X_conv_train = torch.tensor(X_conv_train.reshape((int(args.test_block / args.batch_size), args.batch_size, args.block_len, 2)))
    print(X_conv_train.shape)
    print(Y_train.shape)
    return X_conv_train.float(), Y_train.float()

def _train_epoch(data_x, data_y, model, criterion, optimizer, hidden1, hidden2):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    
    for X, y in zip(data_x, data_y):
        '''
        X should be provided in shape (batch_size, seq_len, feature_size)
        '''
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output, hidden1, hidden2 = model(X.float(), hidden1, hidden2)
        hidden1.detach_()
        hidden2.detach_()
        loss = criterion(output, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
        optimizer.step()
        
        
    return hidden1, hidden2

def _evaluate_help(t_x, t_y, model, criterion, h1, h2):
    running_loss = []
    ber = []
    bler = []
    y_true, y_pred = [], []
    for X, y in zip(t_x, t_y):
        with torch.no_grad():
            output, _, _ = model(X.float(), h1, h2)
            y_true.append(y)
            y_pred.append(output)
            #total += y.size(0)
            #correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
            decoded_bits = np.round(output)
            ber_err_rate  = sum(sum(sum(abs(decoded_bits-y))))*1.0/(y.shape[0]*y.shape[1])
            tp0 = (abs(decoded_bits-y)).reshape([y.shape[0],y.shape[1]]).cpu().numpy()
            bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(y.shape[0])
            ber.append(ber_err_rate)
            bler.append(bler_err_rate)
    return np.mean(running_loss), np.mean(ber), np.mean(bler)

def _evaluate_overall_performance(args, test_x, test_y, model, criterion, hidden1, hidden2):
    hidden1_p = hidden1
    hidden2_p = hidden2
    
    snr_start = args.snr_test_start
    snr_stop  = args.snr_test_end
    snr_points = 10
    
    SNR_dB_start_Eb = snr_start
    SNR_dB_stop_Eb = snr_stop
    SNR_points = snr_points
    
    snr_interval = (SNR_dB_stop_Eb - SNR_dB_start_Eb)* 1.0 /  (SNR_points-1)
    SNRS_dB = [snr_interval* item + SNR_dB_start_Eb for item in range(SNR_points)]
    
    stats = []
    for idx, snr_db in enumerate(SNRS_dB):
        x = channel(test_x, snr_db, snr_db)
        value, ber, bler = _evaluate_help(x, test_y, model, criterion, hidden1_p, hidden2_p)
        stats.append([value, ber, bler])
    stats = np.array(stats)
    
    print(stats)
    
    ber, _ = conv_decode_bench(args)
    bench_ber, = plt.plot(SNRS_dB, ber, '-')
    neural_ber, = plt.plot(SNRS_dB, stats[:, 1], '--')
    
    #bler, = plt.plot(SNRS_dB, stats[:, 2], '--')
    plot_lines = [bench_ber, neural_ber]
    plt.legend(plot_lines, ["viterbi", "neural"])
    plt.xlabel("SNR")
    plt.yscale("log")
    plt.savefig("overall_performance_graph")
    plt.close()
    
    print(stats)
    f = open("data.txt", "a")
    for t in range(3):
        for x in range(len(SNRS_dB)):
            f.write(str(stats[x, t]))
            f.write(" ")
        f.write("\n")  
    f.close()
            
    
def _evaluate_epoch(train_x, train_y, test_x, test_y, model, criterion, train_stats, test_stats, hidden1, hidden2):
    """
    Evaluates the `model` on the train and validation set.
    """
    
    hidden1_p = hidden1
    hidden2_p = hidden2
    
    train_value_loss, train_ber_loss, train_bler_loss = _evaluate_help(train_x, train_y, model, criterion, hidden1_p, hidden2_p)
    test_value_loss, test_ber_loss, test_bler_loss = _evaluate_help(test_x, test_y, model, criterion, hidden1_p, hidden2_p)
    
    train_stats.append([train_value_loss, train_ber_loss, train_bler_loss])
    test_stats.append([test_value_loss, test_ber_loss, test_bler_loss])
    
def snr_db2sigma(train_snr):
    block_len    = 100
    train_snr_Es = train_snr + 10*np.log10(float(block_len)/float(2*block_len))
    sigma_snr    = np.sqrt(1/(2*10**(float(train_snr_Es)/float(10))))
    return sigma_snr

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block', type=int, default=12000)
    parser.add_argument('-test_block', type=int, default=1000)
    parser.add_argument('-block_len', type=int, default=100)
    #parser.add_argument('-test_ratio',  type=int, default=10)

    #parser.add_argument('-num_Dec_layer',  type=int, default=2)
    
    '''
    intermediate state number
    '''
    parser.add_argument('-num_Dec_unit',  type=int, default=400)

    parser.add_argument('-rnn_setup', choices = ['lstm', 'gru'], default = 'gru')

    parser.add_argument('-batch_size',  type=int, default=200)
    parser.add_argument('-learning_rate',  type=float, default=0.001)
    parser.add_argument('-num_epoch',  type=int, default=20)

    parser.add_argument('-code_rate',  type=int, default=2)

    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)
    parser.add_argument('-M',  type=int, default=2, help="Number of delay elements in the convolutional encoder")

    parser.add_argument('-loss', choices = ['binary_crossentropy', 'mean_squared_error'], default = 'mean_squared_error')

    parser.add_argument('-train_channel_low', type=float, default=0.0)
    parser.add_argument('-train_channel_high', type=float, default=8.0)
    parser.add_argument('-test_channel_low', type=float, default=6.0)
    parser.add_argument('-test_channel_high', type=float, default=6.0)
    
    parser.add_argument('-snr_test_start', type=float, default=-3.0)
    parser.add_argument('-snr_test_end', type=float, default=6.0)
    parser.add_argument('-snr_points', type=int, default=10)
    
    parser.add_argument('-noise_type',        choices = ['awgn', 't-dist','hyeji_bursty'], default='awgn')
    parser.add_argument('-radar_power',       type=float, default=20.0)
    parser.add_argument('-radar_prob',        type=float, default=0.05)
    parser.add_argument('-radar_denoise_thd', type=float, default=10.0)
    parser.add_argument('-v',                 type=int,   default=3)
    
    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    parser.add_argument('-Dec_weight', type=str, default='default')
    parser.add_argument('-num_cpu', type=int, default=1)

    args = parser.parse_args()
    print(args)

    print('[ID]', args.id)
    return args

def channel(x, db_low, db_high):
    print('training with noise snr db', db_low, db_high)
    noise_sigma_low =  snr_db2sigma(db_low) # 0dB
    noise_sigma_high =  snr_db2sigma(db_high) # 0dB
    noise_sigma=np.random.uniform(
        high=noise_sigma_high,
        low=noise_sigma_low,
        size=x.shape,
    )

    return x+ noise_sigma*np.random.normal(loc=0., scale=1.0, size = x.shape)   #need to include space for different snrs

def plot_stats(stats, name):
    stats = np.array(stats)
    length = len(stats[:, 0])
    
    #train, = plt.plot(range(length), stats[:, 0], '-')
    ber, = plt.plot(range(length), stats[:, 1], '--')
    #bler, = plt.plot(range(length), stats[:, 2], '--')
    plot_lines = [ber]
    plt.legend(plot_lines, ["ber"])
    plt.xlabel("num_of_epoch")
    plt.yscale("log")
    plt.savefig(name + " graph")
    plt.close()
    
    
def main():
    args = get_args()
    model = RNNGRU(args.code_rate, args.num_Dec_unit, args.block_len)
    
    hidden1 = model.init_hidden(args.batch_size)
    hidden2 = model.init_hidden(args.batch_size)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

    train_x, train_y = _generate_training_data(args)
    test_x, test_y = _generate_test_data(args)
    
    '''
    train_x should be in shape (num_of_batch, batch_size, seq_len, feature_size)
    '''
    train_stats = []
    test_stats = []
    #_evaluate_overall_performance(args, test_x, test_y, model, criterion, hidden1, hidden2)
    
    for epoch in range(0, args.num_epoch):
        print("epoch" , epoch, ":")
        # generate noise
        train_x_noise = channel(train_x, args.train_channel_low, args.train_channel_high)
        test_x_noise = channel(test_x, args.test_channel_low, args.test_channel_high)
        
        # Train model
        hidden1, hidden2 = _train_epoch(train_x_noise, train_y, model, criterion, optimizer, hidden1, hidden2)

        # Evaluate model
        _evaluate_epoch(train_x_noise, train_y, test_x_noise, test_y, model, 
                        criterion, train_stats, test_stats, hidden1, hidden2)
        scheduler_lr.step()
        for param_group in optimizer.param_groups:
            print("learning rate: ", param_group['lr'])
        
        plot_stats(train_stats, "training")
        plot_stats(test_stats, "test")
        print(train_stats)
        
    _evaluate_overall_performance(args, test_x, test_y, model, criterion, hidden1, hidden2)

if __name__ == '__main__':
    main()