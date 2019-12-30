import torch
import torch.nn as nn


class DPCL(nn.Module):
    '''
        Implement of Deep Clustering
    '''

    def __init__(self, num_layer=2, nfft=256, hidden_cells=600, emb_D=40, dropout=0.0, bidirectional=True, activation="Tanh"):
        super(DPCL).__init__()
        self.emb_D = emb_D
        self.blstm = nn.LSTM(nfft, hidden_cells, num_layer, batch_first=True,
                             dropout=dropout, bidirectional=bidirectional, nonlinearity=activation)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(torch.nn,activation)()
        self.linear = nn.Linear(nfft*hidden_cells,emb_D)

    def 

