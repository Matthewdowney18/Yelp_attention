import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_sequences_lengths, variable, argmax


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, embeddings, hidden_size, padding_idx,
                 init_idx, max_len, teacher_forcing):
        """
        Sequence-to-sequence model
        :param vocab_size: the size of the vocabulary
        :param embedding_dim: Dimension of the embeddings
        :param hidden_size: The size of the encoder and the decoder
        :param padding_idx: Index of the special pad token
        :param init_idx: Index of the <s> token
        :param max_len: Maximum length of a sentence in tokens
        :param teacher_forcing: Probability of teacher forcing
        """
        super().__init__()

        self.embedding_dim = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.init_idx = init_idx
        self.max_len = max_len
        self.teacher_forcing = teacher_forcing
        self.vocab_size = embeddings.shape[0]

        ##############################
        ### Insert your code below ###
        ##############################

        self.emb2 = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.enc = nn.LSTM(self.embedding_dim, hidden_size, batch_first=True,
                           bidirectional= False)
        self.dec = nn.LSTMCell(self.embedding_dim, hidden_size)
        self.lin = nn.Linear(hidden_size, self.vocab_size)

        ###############################
        ### Insert your code above ####
        ###############################


    def zero_state(self, batch_size):
        """
        Create an initial hidden state of zeros
        :param batch_size: the batch size
        :return: A tuple of two tensors (h and c) of zeros of the shape of (batch_size x hidden_size)
        """

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = 1
        state_shape = (nb_layers, batch_size, self.hidden_size)

        ##############################
        ### Insert your code below ###
        ##############################
        h0 = variable(torch.zeros(state_shape))
        c0 = variable(torch.zeros(state_shape))
        ###############################
        ### Insert your code above ####
        ###############################

        return h0, c0

    def encode_sentence(self, inputs):
        """
        Encode input sentences input a batch of hidden vectors z
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x hidden_size)
        """
        batch_size = inputs.size(0)

        ##############################
        ### Insert your code below ###
        ##############################

        # Get lengths
        lengths = get_sequences_lengths(inputs, masking=self.padding_idx)

        # Sort as required for pack_padded_sequence input
        lengths, indices = torch.sort(lengths, descending=True)
        inputs = inputs[indices]

        # Pack
        inputs = torch.nn.utils.rnn.pack_padded_sequence(self.emb(inputs),
                                                         lengths.data.tolist(),
                                                         batch_first=True)

        # Encode
        hidden, cell = self.zero_state(batch_size)
        _, (hidden, cell) = self.enc(inputs, (hidden, cell))

        _, unsort_ind = torch.sort(indices)
        z = hidden.squeeze(0)[unsort_ind]

        ###############################
        ### Insert your code above ####
        ###############################

        return z

    def decoder_state(self, z):
        """
        Create initial hidden state for the decoder based on the hidden vectors z
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tuple of two tensors (h and c) of size (batch_size x hidden_size)
        """

        batch_size = z.size(0)

        state_shape = (batch_size, self.hidden_size)

        ##############################
        ### Insert your code below ###
        ##############################
        c0 = variable(torch.zeros(state_shape))
        ###############################
        ### Insert your code above ####
        ###############################

        return z, c0

    def decoder_initial_inputs(self, batch_size):
        """
        Create initial input the decoder on the first timestep
        :param inputs: The size of the batch
        :return: A vector of size (batch_size, ) filled with the index of self.init_idx
        """

        ##############################
        ### Insert your code below ###
        ##############################
        inputs = variable(torch.LongTensor(batch_size, ).fill_(self.init_idx))
        ###############################
        ### Insert your code above ####
        ###############################
        return inputs

    def decode_sentence(self, z, targets=None):
        """
        Decode the tranlation of the input sentences based in the hidden vectors z and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        batch_size = z.size(0)

        ##############################
        ### Insert your code below ###
        ##############################
        # c is the prev
        z, decoder_state = self.decoder_state(z)

        x_i = self.decoder_initial_inputs(batch_size)

        outputs = []
        for i in range(self.max_len):
            embedded = self.emb(x_i)

            z, decoder_state = self.dec(
                embedded, (z, decoder_state))

            output = self.lin(z)
            if targets is not None and i < len(targets):
                x_i = targets[:, i]
            else:
                x_i = torch.multinomial(F.softmax(output, dim=-1), 1).squeeze(-1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        ###############################
        ### Insert your code above ####
        ###############################

        return outputs

    def forward(self, inputs, targets=None):
        """
        Perform the forward pass of the network and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        if self.training and np.random.rand() < self.teacher_forcing:
            targets = inputs
        else:
            targets = None

        z = self.encode_sentence(inputs)
        outputs = self.decode_sentence(z, targets)
        return outputs


#with attention

class Seq2SeqModelAttention(torch.nn.Module):
    def __init__(self, embeddings, hidden_size, padding_idx,
                 init_idx, max_len, teacher_forcing):
        """
        Sequence-to-sequence model
        :param vocab_size: the size of the vocabulary
        :param embedding_dim: Dimension of the embeddings
        :param hidden_size: The size of the encoder and the decoder
        :param padding_idx: Index of the special pad token
        :param init_idx: Index of the <s> token
        :param max_len: Maximum length of a sentence in tokens
        :param teacher_forcing: Probability of teacher forcing
        """
        super().__init__()

        self.embedding_dim = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.init_idx = init_idx
        self.max_len = max_len
        self.teacher_forcing = teacher_forcing
        self.vocab_size = embeddings.shape[0]

        ##############################
        ### Insert your code below ###
        ##############################
        self.attn = nn.Linear(self.hidden_size + self.embedding_dim, self.max_len)
        self.attn_combine = nn.Linear(self.hidden_size +self.embedding_dim,
                                      self.hidden_size)

        #self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.enc = nn.LSTM(self.embedding_dim, hidden_size, batch_first=True,
                           bidirectional= False)
        self.dec = nn.LSTMCell(self.hidden_size, hidden_size)
        self.lin = nn.Linear(hidden_size, self.vocab_size)

        ###############################
        ### Insert your code above ####
        ###############################


    def zero_state(self, batch_size):
        """
        Create an initial hidden state of zeros
        :param batch_size: the batch size
        :return: A tuple of two tensors (h and c) of zeros of the shape of (batch_size x hidden_size)
        """

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = 1
        state_shape = (nb_layers, batch_size, self.hidden_size)

        ##############################
        ### Insert your code below ###
        ##############################
        h0 = variable(torch.zeros(state_shape))
        c0 = variable(torch.zeros(state_shape))
        ###############################
        ### Insert your code above ####
        ###############################

        return h0, c0

    def encode_sentence(self, inputs):
        """
        Encode input sentences input a batch of hidden vectors z
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x hidden_size)
        """
        batch_size = inputs.size(0)

        ##############################
        ### Insert your code below ###
        ##############################

        # Get lengths
        lengths = get_sequences_lengths(inputs, masking=self.padding_idx)

        # Sort as required for pack_padded_sequence input
        lengths, indices = torch.sort(lengths, descending=True)
        inputs = inputs[indices]

        lengths = lengths.data.tolist()

        # Pack
        inputs = torch.nn.utils.rnn.pack_padded_sequence(
            self.emb2(inputs), lengths, batch_first=True)


        # Encode
        hidden, cell = self.zero_state(batch_size)
        output, (hidden, cell) = self.enc(inputs, (hidden, cell))

        output = torch.nn.utils.rnn.pad_packed_sequence(
            output, total_length=self.max_len)[0]

        _, unsort_ind = torch.sort(indices)
        z = output[:,unsort_ind]
        z1 = hidden.squeeze(0)[unsort_ind]

        ###############################
        ### Insert your code above ####
        ###############################

        return z.view(batch_size, self.max_len, self.hidden_size), z1

    def decoder_state(self, z):
        """
        Create initial hidden state for the decoder based on the hidden vectors z
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tuple of two tensors (h and c) of size (batch_size x hidden_size)
        """

        batch_size = z.size(0)

        state_shape = (batch_size, self.hidden_size)

        ##############################
        ### Insert your code below ###
        ##############################
        c0 = variable(torch.zeros(state_shape))
        ###############################
        ### Insert your code above ####
        ###############################

        return z, c0

    def decoder_initial_inputs(self, batch_size):
        """
        Create initial input the decoder on the first timestep
        :param inputs: The size of the batch
        :return: A vector of size (batch_size, ) filled with the index of self.init_idx
        """

        ##############################
        ### Insert your code below ###
        ##############################
        inputs = variable(torch.LongTensor(batch_size, ).fill_(self.init_idx))
        ###############################
        ### Insert your code above ####
        ###############################
        return inputs

    def decode_sentence(self, encoder_outputs, decoder_hidden,targets=None):
        """
        Decode the tranlation of the input sentences based in the hidden vectors z and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        batch_size = encoder_outputs.size(0)

        ##############################
        ### Insert your code below ###
        ##############################
        # c is the prev
        encoder_outputs, decoder_state = self.decoder_state(encoder_outputs)

        x_i = self.decoder_initial_inputs(batch_size)

        outputs = []
        for i in range(self.max_len):
            embedded = self.emb(x_i).view(1, batch_size,-1)
            attn_weights = self.attn(torch.cat((embedded[0], decoder_state), 1))

            attn_weights = F.softmax(attn_weights, dim=1)

            attn_weights = attn_weights.view(batch_size, 1, self.max_len)

            attn_applied = torch.bmm(attn_weights, encoder_outputs).view(batch_size,
                                                                         self.hidden_size)

            output = torch.cat((embedded[0], attn_applied), 1)
            output = self.attn_combine(output)

            decoder_hidden, decoder_state = self.dec(
                output, (decoder_hidden, decoder_state))

            output = self.lin(decoder_hidden)
            if targets is not None and i < len(targets):
                x_i = targets[:, i]
            else:
                x_i = torch.multinomial(F.softmax(output, dim=-1), 1).squeeze(-1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        ###############################
        ### Insert your code above ####
        ###############################

        return outputs

    def forward(self, inputs, targets=None):
        """
        Perform the forward pass of the network and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        if self.training and np.random.rand() < self.teacher_forcing:
            targets = inputs
        else:
            targets = None

        z, z1 = self.encode_sentence(inputs)
        outputs = self.decode_sentence(z, z1, targets)
        return outputs