from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import gym
import os
import ray
from gym.spaces import Discrete, Box
# from sklearn.preprocessing import normalize
# from configs.vars import *
from configs.functions import init_data, get_dataset
from env.StockTradingEnv import StockTradingEnv
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env
# from configs.vars import WALLET_FIRST_SYMBOL, WALLET_SECOND_SYMBOL

if __name__ == "__main__":
    df = pd.read_csv('./datasets/bot_train_ETHBTC_700_day.csv')
    df = df.sort_values('Date')
    register_env("StockTradingEnv-test", lambda config: StockTradingEnv(config))
    ray.init()
    run_experiments({
        "test3_stock_visualization": {
            "run": "PPO",
            "env": "StockTradingEnv-test",
            "stop": {
                "timesteps_total": 1e5, #1e6 = 1M
            },
            "checkpoint_freq": 50,
            "checkpoint_at_end": True,
            "config": {
                "lr": grid_search([
                    1e-4
                    # 1e-6
                ]),
                "num_workers": 2,  # parallelism
                'observation_filter': 'MeanStdFilter',
                "vf_clip_param": 100000.0, # used to trade BTCUSDT
                "env_config": {
                    'df': df
                }
            }
        }
    })