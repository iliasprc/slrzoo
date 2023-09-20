import torch
import torch.nn as nn


class Conv1D(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=-1):
        super(Conv1D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(self.input_size, self.hidden_size, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, ceil_mode=False)
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2, ceil_mode=False)
        if self.num_classes != -1:
            self.fc3 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        '''
            input: (B, C, T)
            output: (B, num_classes, (T-12)/4 )
        '''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        if self.num_classes != -1:
            x = self.fc3(x.transpose(1, 2)).transpose(1, 2)
        return x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, dropout=0.3, bidirectional=True, rnn_type='LSTM',
                 num_classes=-1):
        super(EncoderRNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.input_size = input_size

        self.rnn_type = rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        self.num_classes = num_classes
        if self.num_classes != -1:
            output_size = self.hidden_size * self.num_directions
            # self.dropout = nn.Dropout()
            self.final_fc = nn.Linear(output_size, self.num_classes)

        # for name, param in self.rnn.named_parameters():
        #     if name[:6] == 'weight':
        #         nn.init.orthogonal_(param)

    def forward(self, src_feats, src_lens, hidden=None):
        """
        Args:
            - src_feats: (max_src_len, batch_size, D)
            - src_lens: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """

        # (max_src_len, batch_size, D)
        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens)

        # rnn(gru) returns:
        # - packed_outputs: shape same as packed_emb
        # - hidden: (num_layers * num_directions, batch_size, hidden_size)
        if hidden != None and self.rnn_type == 'LSTM':
            half = int(hidden.size(0) / 2)
            hidden = (hidden[:half], hidden[half:])
        packed_outputs, hidden = self.rnn(packed_emb, hidden)

        # outputs: (max_src_len, batch_size, hidden_size * num_directions)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size) 
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        if isinstance(hidden, tuple):
            hidden = torch.cat(hidden, 0)

        if self.num_classes != -1:
            # outputs = self.dropout(outputs)
            outputs = self.final_fc(outputs)

        return outputs, hidden

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden


if __name__ == "__main__":
    import torch

    model = Conv1D(1024, 1024, 100)
    x = torch.zeros(2, 1024, 20)
    x = model(x)
    print(x.shape)
