import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from buffer import ReplayBuffer
from models import ModelActor, ModelCritic

GAMMA = 0.99
GAE_LAMBDA = 0.95
TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64
TEST_ITERS = 100000
HID_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Play_Agent:
    def __init__(self, env):
        # Торговая среда
        self.env = env

        state_dim = self.env.observation_space.shape[1]
        self.action_size = self.env.action_space.shape[0]

        # Модели Актора и Критика
        self.model_act = ModelActor(state_dim, self.action_size).to(DEVICE)
        self.model_crt = ModelCritic(state_dim).to(DEVICE)

        self.opt_act = optim.Adam(self.model_act.parameters(), lr=LEARNING_RATE_ACTOR)
        self.opt_crt = optim.Adam(self.model_crt.parameters(), lr=LEARNING_RATE_CRITIC)

        # Буффер памяти
        self.memory = ReplayBuffer()

    @torch.no_grad()
    def agent(self, state, hidden_state):
        mu, hidden_state_out = self.model_act(state, hidden_state)
        logstd = self.model_act.logstd
        actions = mu.cpu() + torch.exp(logstd).cpu() * torch.randn(size=logstd.shape,
                                                                   dtype=torch.float32)
        action = torch.clip(actions, min=-1, max=1)
        return action, mu, hidden_state_out

    def calc_logprob(self, mu_v, logstd_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
        return p1 + p2

    def calc_GAE(self, traj_states, rewards, done_v):

        last_gae = 0
        result_adv = []
        result_ref = []

        values = self.calculation_critic_nn(traj_states)

        for val, next_val, reward, done in zip(reversed(values[:-1]), reversed(values[1:]),
                                               reversed(rewards[:-1]), reversed(done_v[:-1])):
            if done:
                delta = reward - val
                last_gae = delta
            else:
                delta = reward + GAMMA * next_val - val
                last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(list(reversed(result_adv)))
        ref_v = torch.FloatTensor(list(reversed(result_ref)))
        return adv_v, ref_v

    def calculation_critic_nn(self, state, batch_crt: int = 32):
        # Расчет значения нейросети Критика
        result = []
        with torch.no_grad():
            for i in range(0, len(self.memory), batch_crt):
                state_ = state[i:i + batch_crt]
                hc = self.model_crt.update_hidden_state(state_.size(0))
                result.append(
                    self.model_crt(state_, hc)
                )
        return torch.cat(result)

    def update_model(self):
        # Обновление весов модели
        traj_states_v = self.memory['obs']
        traj_actions_v = self.memory['action']
        traj_done_v = self.memory['done']
        traj_reward_v = self.memory['reward']
        traj_mu_v = self.memory['mu_v']
        traj_hid_h, traj_hid_c = self.memory['hid_lstm']

        traj_adv_v, traj_ref_v = self.calc_GAE(traj_states_v, traj_reward_v, traj_done_v)
        traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)
        old_logprob_v = self.calc_logprob(traj_mu_v, self.model_act.logstd, traj_actions_v)

        traj_states_v = traj_states_v[:-1]
        traj_actions_v = traj_actions_v[:-1]
        old_logprob_v = old_logprob_v[:-1]

        for epoch in range(PPO_EPOCHES):

            for batch_ofs in range(0, len(self.memory), PPO_BATCH_SIZE):
                states_v = traj_states_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]
                actions_v = traj_actions_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]
                batch_adv_v = traj_adv_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE].unsqueeze(-1)
                batch_ref_v = traj_ref_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE].unsqueeze(-1)
                batch_old_logprob_v = old_logprob_v[batch_ofs:batch_ofs + PPO_BATCH_SIZE]
                batch_hid_h = traj_hid_h[:, batch_ofs:batch_ofs + PPO_BATCH_SIZE, :]
                batch_hid_c = traj_hid_c[:, batch_ofs:batch_ofs + PPO_BATCH_SIZE, :]

                if states_v.size(0) == 0:
                    continue

                # Обновление критика
                self.opt_crt.zero_grad()
                hc = self.model_crt.update_hidden_state(states_v.size(0))
                value_v = self.model_crt(states_v, hc)
                loss_value_v = F.mse_loss(value_v, batch_ref_v.to(DEVICE))
                loss_value_v.backward()
                self.opt_crt.step()

                # Обновление актора
                self.opt_act.zero_grad()
                mu_v, _ = self.model_act(states_v, (batch_hid_h, batch_hid_c))
                batch_new_logprob_v = self.calc_logprob(mu_v, self.model_act.logstd, actions_v)
                ratio_v = torch.exp(batch_new_logprob_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v.to(DEVICE) * ratio_v
                clipped_surr_v = batch_adv_v.to(DEVICE) * torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                loss_policy_v.backward(retain_graph=True)
                self.opt_act.step()

    def play_episod(self, episodes: int) -> None:
        # Средняя награда
        mean_reward = deque(maxlen=100)
        # Общее количество сделанных шагов
        total_steps = 0

        for episod in range(episodes):
            obs = torch.FloatTensor(self.env.reset()).unsqueeze(0).to(DEVICE)
            # Награда и кол-во шагов за эпизод
            rewards = 0
            step = 0
            # Инициализация начального состояния LSTM
            hid_lstm_in = self.model_act.update_hidden_state(1)

            while True:
                action, mu, hid_lstm_out = self.agent(obs, hid_lstm_in)

                obs_new, reward, done, _ = self.env.step(action[0].numpy())
                obs_new = torch.FloatTensor(obs_new).unsqueeze(0).to(DEVICE)

                # Добавление полученных данных в буфер
                self.memory.add((obs, action, float(reward), obs_new, done, mu, hid_lstm_in))

                rewards += float(reward)
                step += 1
                total_steps += 1
                obs = obs_new
                hid_lstm_in = hid_lstm_out

                if done:
                    mean_reward.append(rewards)
                    print('Эпизод:', episod + 1, "Награда:", round(rewards, 2), "Награда сред.:",
                          round(np.mean(mean_reward), 2), "Шагов за эпизод:", step,
                          "Шагов всего:", total_steps)
                    break

                if total_steps % TRAJECTORY_SIZE == 0:
                    self.update_model()
                    self.memory.clear()
