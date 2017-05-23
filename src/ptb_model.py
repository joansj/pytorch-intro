import sys
import torch

########################################################################################################################
# Basic RNN language model
########################################################################################################################

class BasicRNNLM(torch.nn.Module):

    def __init__(self,vocabulary_size):
        super(BasicRNNLM,self).__init__()

        # Configuration of our model
        self.num_layers=2
        embedding_size=650
        self.hidden_size=650
        dropout_prob=0.5

        # Define embedding layer
        self.embed=torch.nn.Embedding(vocabulary_size,embedding_size)

        # Define LSTM
        self.lstm=torch.nn.LSTM(embedding_size,self.hidden_size,self.num_layers,dropout=dropout_prob,batch_first=True)

        # Define dropout
        self.drop=torch.nn.Dropout(dropout_prob)

        # Define output layer
        self.fc=torch.nn.Linear(self.hidden_size,vocabulary_size)

        # Init weights
        init_range=0.1
        self.embed.weight.data.uniform_(-init_range,init_range)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-init_range,init_range)

        return

    def forward(self,x,h):
        # Apply embedding (encoding)
        y=self.embed(x)
        # Run LSTM
        y=self.drop(y)
        y,h=self.lstm(y,h)
        y=self.drop(y)
        # Reshape
        y=y.contiguous().view(-1,self.hidden_size)
        # Fully-connected (decoding)
        y=self.fc(y)
        # Return prediction and states
        return y,h

    def get_initial_states(self,batch_size):
        # Set initial hidden and memory states to 0
        return (torch.autograd.Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)).cuda(),
                torch.autograd.Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)).cuda())

    def detach(self,h):
        # Detach returns a new variable, decoupled from the current computation graph
        return h[0].detach(),h[1].detach()

########################################################################################################################
