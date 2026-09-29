import torch
import torch.nn as nn

class MultiValueLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3, output_size=12, pred_len = 1):
        super(MultiValueLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        # Initialize hidden and cell states
        # if hn is None and cn is None:
        hn = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        cn = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (hn, cn))

        # Take last time step output
        out = out[:, -self.pred_len, :]

        # Fully connected layer
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.unsqueeze(1)
        return out