#####################################################################################
# This code was adapted from:                                                       #
# https://github.com/deeplearningathome/pytorch-language-model/blob/master/lm.py    #
#####################################################################################
import torch.nn as nn

class LSTM(nn.Module):
    """
    This class contains the LSTM model used in the project.
    """

    def __init__(self, vocab_size, batch_size, embedding_size, hidden_size, num_layers, dropout_rate, num_step):
        super(LSTM, self).__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_step = num_step
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                num_layers=num_layers, dropout=dropout_rate)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_size))


    def forward(self, input, hidden):
        emb = self.drop(self.word_embeddings(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        logits = self.fc(output.view(-1, self.hidden_size))
        return logits.view(output.shape[0], output.shape[1], self.vocab_size), hidden
