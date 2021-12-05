import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint

import sys
sys.path.append("../FinRL-Library")

import itertools

import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

print("============== CONFIG DATES ===========")
START_TRAINING_DATE = '2010-01-01'
END_TRAINING_DATE = '2019-01-01'
START_TRADING_DATE = '2019-01-01'
END_TRADING_DATE = '2021-01-01'

print('START_TRAINING_DATE :'+START_TRAINING_DATE)
print('END_TRAINING_DATE :'+END_TRAINING_DATE)
print('START_TRADING_DATE :'+START_TRADING_DATE)
print('END_TRADING_DATE :'+END_TRADING_DATE)

SEED = 41
np.random.seed(SEED)


print("==============Prepare Training Data===========")
# df = pd.read_csv('JII_data_google_20210613.csv')

# fe = FeatureEngineer(
#                     use_technical_indicator=True,
#                     tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
#                     use_turbulence=True,
#                     user_defined_feature = False)

# processed = fe.preprocess_data(df)

# list_ticker = processed["tic"].unique().tolist()
# list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
# combination = list(itertools.product(list_date,list_ticker))

# processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
# processed_full = processed_full[processed_full['date'].isin(processed['date'])]
# processed_full = processed_full.sort_values(['tic','date']).fillna(method='ffill')

# processed_full = processed_full.sort_values(['date','tic'])

# processed_full.to_csv('JII_data_processed_20210613.csv', index=False)

processed_full = pd.read_csv('JII_data_processed_20210613.csv')
processed_full = processed_full.sort_values(['date','tic'])

print("==============Set up Environment==========")

train = data_split(processed_full, START_TRAINING_DATE, END_TRAINING_DATE)

stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100, 
    "initial_amount": 150_000_000/100, #Since in Indonesia the minimum number of shares per trx is 100, then we scaled the initial amount by dividing it with 100 
    "buy_cost_pct": 0.0019, #IPOT has 0.19% buy cost
    "sell_cost_pct": 0.0029, #IPOT has 0.29% sell cost
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4,
    "print_verbosity":5
    
}

A2C_PARAMS = {
    "n_steps": 5, 
    "ent_coef": 0.01, 
    # "learning_rate": 0.0007,
    "learning_rate": 0.0005,
    "seed": SEED,
}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0001,
    "batch_size": 128,
    "seed": SEED,
}
DDPG_PARAMS = {
    "batch_size": 128, 
    "buffer_size": 50000, 
    "learning_rate": 0.0007,
    "seed": SEED,
}

# TRAINING ENV
e_train_gym = StockTradingEnv(df = train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

agent = DRLAgent(env = env_train)

# TRADING ENV
data_turbulence = processed_full[(processed_full.date<END_TRADING_DATE) & (processed_full.date>=START_TRAINING_DATE)]
insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])

# turbulence_threshold = np.quantile(insample_turbulence.turbulence.values,0.98)
turbulence_threshold = 100
# print("Turbulence threshold: "+turbulence_threshold)

trade = data_split(processed_full, START_TRADING_DATE, END_TRADING_DATE)
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = turbulence_threshold, **env_kwargs)



print("============== Train Agent A2C ==========")
model_a2c = agent.get_model("a2c",model_kwargs = A2C_PARAMS)
trained_a2c = agent.train_model(model=model_a2c, 
                             tb_log_name='a2c',
                             total_timesteps=100000)

trained_a2c.save(f"{config.TRAINED_MODEL_DIR}/A2C_100K_20210815_google")

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_a2c, 
    environment = e_trade_gym)
df_account_value.tail()

now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
df_account_value.to_csv("./"+config.RESULTS_DIR+"/df_account_value_"+now+'_A2C.csv')
df_actions.to_csv("./"+config.RESULTS_DIR+"/df_actions_"+now+'_A2C.csv')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'_A2C.csv')

print("============== Train Agent DDPG ==========")
model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)
trained_ddpg = agent.train_model(model=model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=50000)
                            
trained_ddpg.save(f"{config.TRAINED_MODEL_DIR}/DDPG_50K_20210815_google")

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ddpg, 
    environment = e_trade_gym)
df_account_value.tail()

now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
df_account_value.to_csv("./"+config.RESULTS_DIR+"/df_account_value_"+now+'_DDPG.csv')
df_actions.to_csv("./"+config.RESULTS_DIR+"/df_actions_"+now+'_DDPG.csv')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'_DDPG.csv')

print("============== Train Agent PPO ==========")

model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=100_000)

trained_ppo.save(f"{config.TRAINED_MODEL_DIR}/PPO_100K_20210815_google")

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ppo, 
    environment = e_trade_gym)
df_account_value.tail()

now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
df_account_value.to_csv("./"+config.RESULTS_DIR+"/df_account_value_"+now+'_PPO.csv')
df_actions.to_csv("./"+config.RESULTS_DIR+"/df_actions_"+now+'_PPO.csv')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'_PPO.csv')

print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^JKLQ45", 
        start = START_TRADING_DATE,
        end = END_TRADING_DATE)

stats = backtest_stats(baseline_df, value_col_name = 'close')
