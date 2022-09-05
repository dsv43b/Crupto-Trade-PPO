import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OneDataStep:
    def __init__(self, obs, action, reward, obs_new, done, mu_v, hid_lstm):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.obs_new = obs_new
        self.done = done
        self.mu_v = mu_v
        self.hid_lstm = hid_lstm


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add(self, data: OneDataStep) -> None:
        self.buffer.append(OneDataStep(*data))

    def clear(self):
        self.buffer.clear()


    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        # obs, action, reward, obs_new, done, mu_v, hid_lstm_h, hid_lstm_c (self.hid_lstm)
        if item in ('reward', 'done'):
            return torch.FloatTensor(
                list(map(lambda x: getattr(x, item), self.buffer))
            ).view(-1, 1).to(DEVICE)
        elif item == 'hid_lstm':
            return (torch.cat(list(map(lambda x: x.hid_lstm[0], self.buffer)), dim=1).to(DEVICE),
                    torch.cat(list(map(lambda x: x.hid_lstm[1], self.buffer)), dim=1).to(DEVICE)
                    )
        else:
            return torch.cat(list(map(lambda x: getattr(x, item), self.buffer))).to(DEVICE)
