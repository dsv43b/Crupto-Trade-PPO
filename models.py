from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size, num_layers_lstm=1):
        super(ModelActor, self).__init__()
        self.num_layers_lstm = num_layers_lstm

        self.before_lstm = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(256, 256, self.num_layers_lstm, batch_first=True)
        self.after_lstm = nn.Sequential(
            nn.Linear(256, act_size),
            nn.Tanh()
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x, hidden_in):
        batch = x.size(0)
        # h0_in, c0_in = hidden_in
        x = self.before_lstm(x)

        _, hidden_out = self.lstm(x, hidden_in)

        h_o = hidden_out[0].reshape(batch, -1)
        return self.after_lstm(h_o), hidden_out

    def update_hidden_state(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return torch.zeros([self.num_layers_lstm, batch_size, 256], device=DEVICE), \
               torch.zeros([self.num_layers_lstm, batch_size, 256], device=DEVICE)


class ModelCritic(nn.Module):
    def __init__(self, obs_size, num_layers_lstm=1):
        super(ModelCritic, self).__init__()

        self.num_layers_lstm = num_layers_lstm
        self.before_lstm = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(256, 256, self.num_layers_lstm, batch_first=True)

        self.after_lstm = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x, hidden_in):
        batch = x.size(0)
        x = self.before_lstm(x)

        _, hidden_out = self.lstm(x, hidden_in)
        h_o = hidden_out[0].reshape(batch, -1)
        return self.after_lstm(h_o)

    def update_hidden_state(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return torch.zeros([self.num_layers_lstm, batch_size, 256], device=DEVICE), \
               torch.zeros([self.num_layers_lstm, batch_size, 256], device=DEVICE)
