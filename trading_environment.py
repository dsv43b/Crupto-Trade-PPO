from typing import List, Tuple

import gym
import numpy as np
import pandas as pd
from gym import spaces


class TradingEnv(gym.Env):

    def __init__(self, data: pd.DataFrame, ceemdan_data: np.array,
                 start_balance: int = 100_000, time_step: int = 96,
                 exchange_commission: float = 0.1, lot_size: int = 10000) -> None:
        super(TradingEnv, self).__init__()

        self.start_balance = start_balance
        self.lot_size = lot_size
        self.time_step = time_step
        self.exchange_commission = exchange_commission
        self.data = data.drop(columns='Full_time').to_numpy()
        data_shape = self.data.shape[1] + ceemdan_data.shape[0] * ceemdan_data.shape[2]
        # Обработанные цены закрытия с помощью ceemdan
        self.ceemdan_data = ceemdan_data
        self.close_price_crupto = data[['Close_1', 'Close_2', 'Close_3']].to_numpy()

        self.action_space = spaces.Box(low=-1., high=1.,
                                       shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(low=.0, high=1.,
                                            shape=(time_step, data_shape), dtype=np.float32)

    def reset(self) -> np.array:
        """
        Сброс параметров на начальные
        """
        # Баланс по каждой валюте USD, BTC, ETH, LTC (абсолютные величины)
        self.currency_balance = np.array([self.start_balance, .0, .0, .0])
        # Относительное значение баланса криптовалюты
        self.relative_balance = np.zeros(3)
        # self.data[:, -4:] = 0  # В этой версии не используется

        # Общий баланс
        self.united_balance = [self.start_balance]
        # Общее кол-во шагов за эпизод
        self.global_step = 0
        return self.__standardization(
            self.data[:self.time_step], self.ceemdan_data[:, self.global_step])

    def __Sortino_ratio(self, risk_free: float = 0.0) -> float:
        """
        Расчет коэф-та Сортино
        """
        if len(self.united_balance) < self.time_step + 1:
            local_start_bln = np.array(self.united_balance[:-1])
            local_bln = np.array(self.united_balance[1:])
        else:
            local_start_bln = np.array(self.united_balance[-self.time_step - 1: -1])
            local_bln = np.array(self.united_balance[-self.time_step:])

        delta_bln = (local_bln - local_start_bln) / local_start_bln
        # Коэффициент Сортино
        SOR = (np.mean(delta_bln) - risk_free) / \
              (np.std(delta_bln[delta_bln < risk_free]) + 1e-6)
        return float(np.clip(np.nan_to_num(SOR), -1, 5))

    def __transaction_calculation(self, execution_order: np.array) -> float:
        """
        Расчет текущего баланса с учетом стоимости всех криптовалют
        """
        # Расчет размера лота для каждой валюты
        base_lots = self.lot_size / self.close_price_crupto[self.global_step + self.time_step - 1]
        # Текущая стоимость криптовалюты
        price = self.close_price_crupto[self.global_step + self.time_step - 1]
        # Стоимость криптовалюты на следующем шаге
        price_next = self.close_price_crupto[self.global_step + self.time_step]
        # Расчитанный лот со знаком направления сделки
        lot = base_lots * execution_order
        # Сумма покупки/продажи в USD
        transaction_size = lot * price
        self.currency_balance[0] -= np.sum(transaction_size + self.__commission_fee(lot, price))
        # Относительное значение
        self.relative_balance += execution_order
        # Действительное кол-во криптовалюты
        self.currency_balance[1:] += lot
        return np.sum(self.currency_balance[1:] * price_next) + self.currency_balance[0]

    def __commission_fee(self, lot_: float, price: float) -> float:
        """
        Расчет комиссии за сделку
        """
        return abs(lot_) * price * self.exchange_commission / 100

    def __standardization(self, data: np.array, ceemdan: np.array) -> np.array:
        """
        Объединение и стандартизация данных с ценами криптовалют и после
        преобразования ceemdan.
        """
        ceemdan = np.swapaxes(ceemdan.reshape(-1, self.time_step), 0, 1)
        data = np.concatenate([data, ceemdan], axis=1)
        data = (data - data.min(axis=0)[np.newaxis, :]) / (data.max(axis=0)[np.newaxis, :] - data.min(axis=0)[np.newaxis, :])
        return np.nan_to_num(data)

    def step(self, action: List[float]) -> Tuple[np.array, float, bool, dict]:
        """
        Входное значение action -> [0.0, 0.0, 0.0].
        Каждое из 3-х значений соответствует трем криптовалютам.
        Если значения больше 0, то покупаем, если меньше 0 то продаем.

        1 lot = 10_000
        """
        # Получение баланса по долларам и криптовалюте
        last_balance_usd = self.currency_balance[0]

        # Вычисление выполняемых действий
        __execution_order = np.clip(action + self.relative_balance, -1, 1)
        execution_order = __execution_order - self.relative_balance

        # Совершение сделок и получение текущего баланса
        current_balance = self.__transaction_calculation(execution_order)
        self.united_balance.append(current_balance)

        # Вознаграждение(изменение баланса)
        PR = (self.united_balance[-1] - self.united_balance[-2]) / self.united_balance[-2]
        # Коэффициент Сортино
        sr_coff = self.__Sortino_ratio()
        reward = PR + sr_coff

        # Флаг завершения эпизода
        done = True if (self.global_step == self.data.shape[0] - self.time_step - 2
                        or self.united_balance[-1] <= 0) else False

        self.global_step += 1

        next_state = self.__standardization(
            self.data[self.global_step:self.global_step + self.time_step],
            self.ceemdan_data[:, self.global_step]
        )
        return next_state, reward, done, {}

    def render(self):
        pass

    def close(self):
        pass
