import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence,pack_sequence,pad_sequence

class DPCL(nn.Module):
    '''
        Implement of Deep Clustering
    '''

    def __init__(self, num_layer=2, nfft=129, hidden_cells=600, emb_D=40, dropout=0.0, bidirectional=True, activation="Tanh"):
        super(DPCL, self).__init__()
        self.emb_D = emb_D
        self.blstm = nn.LSTM(input_size=nfft, hidden_size=hidden_cells,num_layers=num_layer, batch_first=True,
                             dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(torch.nn, activation)()
        self.linear = nn.Linear(2*hidden_cells if bidirectional else hidden_cells, nfft * emb_D)
        self.D = emb_D
    def forward(self, x, is_train=True):
        '''
           input: 
                  for train: B x T x F
                  for test: T x F
           return: 
                  for train: B x TF x D
                  for test: TF x D
        '''
        if not is_train:
            x = torch.unsqueeze(x, 0)
        # B x T x F -> B x T x hidden
        x, _ = self.blstm(x)
        if is_train:
            x,_ = pad_packed_sequence(x,batch_first=True)
        x = self.dropout(x)

        # B x T x hidden -> B x T x FD
        x = self.linear(x)
        x = self.activation(x)

        B = x.shape[0]
        if is_train:
            # B x TF x D
            x = x.view(B,-1,self.D)
        else:
            # B x TF x D -> TF x D
            x = torch.squeeze(x)

        return x
    
if __name__ == "__main__":
    device = torch.device('cuda:0')
    a = torch.randn((11,129))
    #b = torch.randn((22,129))
    #c = torch.randn((33,129))
    train = pack_sequence([a]).to(device)
    net = DPCL().to(device)
    x = net(train)