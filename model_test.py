#
def hello_word():
  print("hello word!")

import torch
import torch.nn as nn
class GRUModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.1):
      """
      Args:
          input_dim (int): Number of input features (D)
          hidden_dim (int): Number of hidden units in GRU
          output_dim (int): Number of output features (D)
          num_layers (int): Number of GRU layers
          dropout (float): Dropout rate for regularization
      """
      super(GRUModel, self).__init__()
      self.hidden_dim = hidden_dim
      self.num_layers = num_layers
      #pre-net
      #self.pre1 = nn.Linear(input_dim, hidden_dim)
      #self.pre2 = nn.Linear(hidden_dim, hidden_dim)
      self.dp = nn.Dropout(p=0.25)
      # GRU layer
      self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
      self.bn_gru = nn.LayerNorm(hidden_dim)
      ## Fully connected layer to map the hidden state to outputù
      self.fc1 = nn.Linear(hidden_dim, hidden_dim)
      self.bn_fc1 = nn.LayerNorm(hidden_dim)
      self.fc_mean =  nn.Linear(hidden_dim, output_dim)
      self.fc_logvar =  nn.Linear(hidden_dim, 6)
      self.relu = nn.ReLU()

  def forward(self, x):
      """
      Args:
          x (tensor): Shape [batch_size, sequence_length, input_dim]

      Returns:
          y_pred (tensor): Shape [batch_size, output_dim]
      """
      # GRU forward pass
      #x1 = self.relu(self.pre1(x))
      #x1 = self.dp(x1)
      #x1 = self.relu(self.pre2(x1))
      #x1 = self.dp(x1)
      _, h_n = self.gru(x)  # h_n is the final hidden state, shape [num_layers, batch_size, hidden_dim]

      # Use the last layer's hidden state for prediction
      #print("last hn", h_n.shape)
      #added residualwith projection
      h_n = h_n[-1] #+ x1[:, -1, :] # Shape [batch_size, hidden_dim]# first dim is GRU stack layers porco dio
      h_n = self.bn_gru(h_n)
      # rete ENCODER 
      # E = CNN1D(X_1min)
      # h_n concatenato a "E"
      #print("take last", h_n.shape)
      h_n = self.relu(h_n)
      h_n = self.fc1(h_n)
      h_n = self.bn_fc1(h_n)
      h_n = self.relu(h_n)
      h_n = self.dp(h_n)
      # Fully connected layer for output prediction
      y_mean = self.fc_mean(h_n)  # Shape [batch_size, output_dim]
      y_logvar = self.fc_logvar(h_n)
      #print(y_pred.shape)
      #print(x.mean(dim=1).shape)
      return y_mean, y_logvar.exp()  # ✅ predizione assoluta

      #return (x[:,-1,:] - y_mean), y_logvar.exp()# res layer
