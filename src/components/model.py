import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        #x = x[:, -1, :]
        x = self.linear(x)
        return x