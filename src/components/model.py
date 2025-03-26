import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
    

class LSTM2(nn.Module):
     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM2, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True )
        self.fc = nn.Linear(hidden_dim, output_dim)

     def forward(self, x):
         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
         c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
         out, _ = self.lstm(x,(h0,c0))
         out = self.fc(out[:,-1,:])
         return out
    

