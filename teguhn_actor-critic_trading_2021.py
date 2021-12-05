import warnings
warnings.filterwarnings("ignore")

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
START_TRADING_DATE = '2020-06-01'
END_TRADING_DATE = '2021-07-09'

print('START_TRADING_DATE :'+START_TRADING_DATE)
print('END_TRADING_DATE :'+END_TRADING_DATE)

TICKER_LIST = config.JII_TICKER

print("==============Prepare Trading Data===========")
df = YahooDownloader(start_date = START_TRADING_DATE,
                     end_date = END_TRADING_DATE,
                     ticker_list = TICKER_LIST).fetch_data()

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=True,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['tic','date']).fillna(method='ffill')

processed_full = processed_full.sort_values(['date','tic'])

# processed_full = pd.read_csv('JII_data_processed_20210613.csv')
# processed_full.to_csv('trading_data_processed_'+START_TRADING_DATE+'_to_'+END_TRADING_DATE+'.csv', index=False)

print("==============Set up Environment==========")

trade = data_split(processed_full, START_TRADING_DATE, END_TRADING_DATE)

stock_dimension = len(trade.tic.unique())
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

# TRADING ENV
data_turbulence = processed_full
insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])

turbulence_threshold = np.quantile(insample_turbulence.turbulence.values,0.98)
print("Turbulence threshold: ", turbulence_threshold)

e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = turbulence_threshold, **env_kwargs)

env_trade, _ = e_trade_gym.get_sb_env()
agent = DRLAgent(env = env_trade)

print("============== Trade Agent A2C ==========")
model_a2c = agent.get_model("a2c")
trained_a2c = model_a2c.load(f"{config.TRAINED_MODEL_DIR}/A2C_100K_20210613_google")

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

print("============== Trade Agent DDPG ==========")
model_ddpg = agent.get_model("ddpg")
                            
trained_ddpg = model_ddpg.load(f"{config.TRAINED_MODEL_DIR}/DDPG_50K_20210613_google")

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

print("============== Trade Agent PPO ==========")

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

trained_ppo = model_ppo.load(f"{config.TRAINED_MODEL_DIR}/PPO_100K_20210613_google")

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

#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^JKLQ45", 
        start = START_TRADING_DATE,
        end = END_TRADING_DATE)

stats = backtest_stats(baseline_df, value_col_name = 'close')
perf_stats = pd.DataFrame(stats)
# perf_stats.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'_LQ45.csv')

baseline_df = get_baseline(
        ticker="^JKII", 
        start = START_TRADING_DATE,
        end = END_TRADING_DATE)

stats = backtest_stats(baseline_df, value_col_name = 'close')
perf_stats = pd.DataFrame(stats)
# perf_stats.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'_JII.csv')
