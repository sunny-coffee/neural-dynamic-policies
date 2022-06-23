import torch
import torch.nn as nn


class Elmo(nn.Module):
    def __init__(self, state_dim, hidden_size, gru_layer_num=3, dropout=0.1, init_w=0.003):
        super(Elmo, self).__init__()
        self.hidden_size = hidden_size
        self.gru_layer_num = gru_layer_num
        self._embedding = nn.Linear(in_features=state_dim, out_features=hidden_size)     
        self.elmo_blocks = nn.ModuleList(
            [ElmoBlock(hidden_size, hidden_size, dropout) for _ in range(gru_layer_num)])
        # for fit
        self._linear = nn.Linear(hidden_size, state_dim)
        self._linear.weight.data.uniform_(-init_w, init_w)
        self._linear.bias.data.uniform_(-init_w, init_w)
        # for predict
        self.gamma = nn.Parameter(torch.FloatTensor(1))
        self.s = nn.Parameter(torch.FloatTensor(gru_layer_num+1))
        # print(self.gamma, self.s)

    def fit(self, posSeq):
        # posSeq = posSeq.permute(1,0,2)
        # print(posSeq.shape)
        outPosSeq = torch.empty(posSeq.shape)
        batch_size = posSeq.shape[0]
        hiddens = torch.empty((batch_size, self.gru_layer_num, self.hidden_size))
        hidden = torch.zeros((batch_size, self.hidden_size), dtype=torch.float32)
        for i in range(posSeq.shape[1]):
            # print(posSeq[:,i,:].shape)
            x = self._embedding(posSeq[:,i,:])
            # print(x.shape)
            for j, block in enumerate(self.elmo_blocks):
                # print(x.shape, hiddens[:,j,:].shape)
                x, current_h = block(x, hidden)
                hidden = current_h
                hiddens[:,j,:] = current_h
            outPosSeq[:,i,:] = self._linear(x)

        return outPosSeq

    def forward(self, input, hiddens):
        next_hiddens = hiddens.new(hiddens.shape)
        x = self._embedding(input)
        ret = self.s[0] * x
        for i, block in enumerate(self.elmo_blocks):
            x, current_h = block(x, hiddens[i])
            next_hiddens[i] = current_h
            ret = ret + self.s[i+1] * current_h
        return self.gamma * ret


class ElmoBlock(nn.Module):
    def __init__(self, hidden_size, norm_size, dropout):

        super(ElmoBlock, self).__init__()
        self._gru = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.norm = nn.LayerNorm(norm_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, h):
        # print(x.shape, h.shape)
        current_h = self._gru(x, h)
        output = x + self.dropout((self.norm(current_h)))
        return output, current_h



# state_dim=2
# hidden_size=8
# seqLen = 4
# batch_size = 3
# n_layers = 3

# elmo = Elmo(state_dim, hidden_size)
# inputSeq = torch.zeros((seqLen,batch_size, 2))
# # print(elmo.fit(inputSeq))
# print(elmo.fit(inputSeq).shape)

# hiddens = torch.zeros((n_layers,batch_size, hidden_size), dtype=torch.float32)
# x = torch.zeros((batch_size, state_dim), dtype=torch.float32)
# print(elmo(x,hiddens).shape)