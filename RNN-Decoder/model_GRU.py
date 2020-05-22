import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RNNGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru_layer1 = nn.GRU(input_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.gru_layer2 = nn.GRU(hidden_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout1 = nn.Dropout(p = 0.3)
        self.dropout2 = nn.Dropout(p = 0.3)
        # The linear layer that maps from hidden state space to tag space
        self.batch_layer1 = nn.BatchNorm1d(seq_len)
        self.batch_layer2 = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(hidden_dim, 1)
        
        
        #self.init_weights()
        nn.init.normal_(self.linear_layer.weight, 0.0, 0.01)
        nn.init.constant_(self.linear_layer.bias, 0.0)
        
    def forward(self, input_seq, hidden1, hidden2):
        intermed, hidden1 = self.gru_layer1(input_seq, hidden1)
        intermed = self.dropout1(intermed)
        intermed = self.batch_layer1(intermed)

        intermed, hidden2 = self.gru_layer2(intermed, hidden2)
        intermed = self.dropout2(intermed)
        intermed = self.batch_layer2(intermed)

        intermed = self.linear_layer(intermed)
        return intermed, hidden1, hidden2
    
    def init_weights(self):
        
        for gru in [self.gru_layer1, self.gru_layer2]:
            #C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)
        for bat in [self.batch_layer1, self.batch_layer2]:
            #C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)
        nn.init.normal_(self.linear_layer.weight, 0.0, 0.01)
        nn.init.constant_(self.linear_layer.bias, 0.0)
        
    def init_hidden(self, batch_size):
        # variable of size [num_layers*num_directions, b_sz, hidden_sz]
        return torch.zeros(2, batch_size, self.hidden_dim//2).float()

