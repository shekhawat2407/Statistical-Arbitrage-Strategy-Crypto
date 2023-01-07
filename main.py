import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import implementation_steps as implemetation
import time
import functions as fn

warnings.filterwarnings("ignore")

# Testing period window
t_start = datetime(2021, 9, 26, 0)
t_end = datetime(2022, 9, 25, 23)
hour_delta = timedelta(0, 0, 0, 0, 0, 1, 0)
delta_t = 1 / 8760
window = 240

# Reading data for top40 coin names
top40_coin_info = pd.read_csv("coin_universe_150K_40.csv", parse_dates=['startTime'])

for i in range (0,top40_coin_info.shape[0]):
    if top40_coin_info.iloc[i,:].isin(['ETH']).sum()==0:
        top40_coin_info.iloc[i,-1]='ETH'
# using tz_localize to remove the timezone information once it is noted
top40_coin_info['startTime'] = pd.to_datetime(top40_coin_info.startTime).dt.tz_localize(None)
# forward filling the coin data
top40_coin_info.ffill(inplace=True)

## Hourly price data for all the coins
all_coin_data = pd.read_csv("coin_all_prices_full_final_version.csv", parse_dates=['startTime'])

# using tz_localize to remove the timezone information once it is noted
all_coin_data['startTime'] = pd.to_datetime(all_coin_data.startTime).dt.tz_localize(None)
all_coin_data.set_index('startTime', inplace=True)
all_coin_data.drop(columns='time', inplace=True)
# Calculating hourly returns for all the coins
all_coin_data_ret = all_coin_data.pct_change(1)
all_coin_data_ret.replace([np.inf, -np.inf], np.nan, inplace=True)

all_coin_data_ret_test=all_coin_data_ret[(all_coin_data_ret.index>=t_start)&(all_coin_data_ret.index<=t_end)]
all_coin_data_ret_test.fillna(0)


## initialising the parameters for the trading signal
sbo = 1.25
sso = 1.25
sbc = 0.75
ssc = 0.5

# Creating data frame to store the portfolio positions in the testing period
Position_Frame=pd.DataFrame(data=0,columns=all_coin_data.columns,index=all_coin_data.index)
Position_Frame=Position_Frame[(Position_Frame.index>=t_start-hour_delta)&(Position_Frame.index<=t_end)]
Signal_Frame=pd.DataFrame(data=0,columns=all_coin_data.columns,index=all_coin_data.index)
Signal_Frame=Signal_Frame[(Signal_Frame.index>=t_start)&(Signal_Frame.index<=t_end)]
Eigen_Portfolio_1=pd.DataFrame(data=0,columns=Signal_Frame.columns,index=Signal_Frame.index)
Eigen_Portfolio_2=pd.DataFrame(data=0,columns=Signal_Frame.columns,index=Signal_Frame.index)
Vector_1=pd.DataFrame(data=0,columns=Signal_Frame.columns,index=Signal_Frame.index)
Vector_2=pd.DataFrame(data=0,columns=Signal_Frame.columns,index=Signal_Frame.index)
BTC_sscore=pd.Series(data=0,index=Signal_Frame.index)
ETH_sscore=pd.Series(data=0,index=Signal_Frame.index)
periods=Signal_Frame.shape[0]

## Implementation of Strategy
index_time = Signal_Frame.index.to_list()
Signal_Frame_T = Signal_Frame.transpose()
Eigen_Portfolio_1_T = Eigen_Portfolio_1.transpose()
Eigen_Portfolio_2_T = Eigen_Portfolio_2.transpose()
Vector_1_T = Vector_1.transpose()
Vector_2_T = Vector_2.transpose()
init_implementation = implemetation.implementation_steps(top40_coin_info, hour_delta, all_coin_data_ret, delta_t,sbo,
                                                         sso, sbc, ssc, t_start, t_end, window)

for i in range(0, periods):
    t_rel = index_time[i]
    Position_Frame, signal, vector1, vector2, factor1, factor2, btc_score, eth_score = \
                                            init_implementation.strategy_implementation(t_rel,Position_Frame)
    Signal_Frame_T[t_rel] = signal
    Eigen_Portfolio_1_T[t_rel] = factor1
    Eigen_Portfolio_2_T[t_rel] = factor2
    Vector_1_T[t_rel] = vector1
    Vector_2_T[t_rel] = vector2
    BTC_sscore.loc[t_rel] = btc_score[0]
    ETH_sscore.loc[t_rel] = eth_score[0]

Signal_Frame = Signal_Frame_T.transpose()


print("Starting Task 1.1")
########################################################################################################
#### Task 1.1: Exporting the Eigen Portfolio weights corresponding to the largest two eigenvalues to CSV
########################################################################################################

Eigen_Portfolio_1=Eigen_Portfolio_1_T.transpose()
Eigen_Portfolio_1.fillna(0,inplace=True)
Eigen_Portfolio_2=Eigen_Portfolio_2_T.transpose()
Eigen_Portfolio_2.fillna(0,inplace=True)

#### Writing to csv file

Eigen_Portfolio_1.to_csv("task1a_1.csv")
Eigen_Portfolio_2.to_csv("task1a_2.csv")


print("Task 1.1 ended. Both Eigen Portfolios have been written to csv")

print("Starting Task 1.2")
####

########################################################################################################
#### Task 1.2: Plotting returns of the Eigen Portfolios, BTC & Ethereum
########################################################################################################
BTC_cumul_returns=(all_coin_data_ret_test['BTC']+1).cumprod() -1
ETH_cumul_returns=(all_coin_data_ret_test['ETH']+1).cumprod() -1
cumulative_returns_factor1=fn.calculate_parameters(Eigen_Portfolio_1, all_coin_data_ret_test = all_coin_data_ret_test, periods = periods)['cumulative_returns']
cumulative_returns_factor2=fn.calculate_parameters(Eigen_Portfolio_2, all_coin_data_ret_test = all_coin_data_ret_test, periods = periods)['cumulative_returns']
cumulative_returns_factor1=cumulative_returns_factor1.rename('Largest Eigen Portfolio',inplace=True).to_frame()
cumulative_returns_factor2=cumulative_returns_factor2.rename('Second Largest Eigen Portfolio',inplace=True).to_frame()
BTC_cumul_returns=BTC_cumul_returns.rename('BTC',inplace=True).to_frame()
ETH_cumul_returns=ETH_cumul_returns.rename('ETH',inplace=True).to_frame()

returns_combined=pd.concat([cumulative_returns_factor1,cumulative_returns_factor2,BTC_cumul_returns,ETH_cumul_returns],axis=1)
fn.plot_graph(returns_combined,'Cumulative Returns',"Time","Returns",0,'Cummulative_Return_Curves_4_assets.png')

print("Task 1.2 ended. Cummulative Return Curve for 4 assets have been saved")
print("Starting Task 2")

########################################################################################################
# Task 2: Plotting the eigen-portfolio weights of the two eigen-portfolios at given hours (2021-09-26 12:00:00 & 2022-04-15 20:00:00)
########################################################################################################


t1 = datetime(2021, 9, 26, 12)
t2 = datetime(2022, 4, 15, 20)


for time in [t1]:
    weights_t_E1 = Eigen_Portfolio_1[Eigen_Portfolio_1.index == time].transpose()
    weights_t_E1 = weights_t_E1[weights_t_E1 != 0].dropna()
    weights_t_E1 = weights_t_E1.squeeze().sort_values(ascending=False)
    fn.plot_graph(weights_t_E1, "Largest Eigen Portfolio Weights at " + str(time), "Coins", "Weights", ticks=1,image_name='time1_eigen_portfolio_1.png')

    weights_t_E2 = Eigen_Portfolio_2[Eigen_Portfolio_2.index == time].transpose()
    weights_t_E2 = weights_t_E2[weights_t_E2 != 0].dropna()
    weights_t_E2 = weights_t_E2.squeeze().sort_values(ascending=False)
    fn.plot_graph(weights_t_E2, "Second Largest Eigen Portfolio Weights at " + str(time), "Coins", "Weights", ticks=1, image_name='time1_eigen_portfolio_2.png')


for time in [t2]:
    weights_t_E1 = Eigen_Portfolio_1[Eigen_Portfolio_1.index == time].transpose()
    weights_t_E1 = weights_t_E1[weights_t_E1 != 0].dropna()
    weights_t_E1 = weights_t_E1.squeeze().sort_values(ascending=False)
    fn.plot_graph(weights_t_E1, "Largest Eigen Portfolio Weights at " + str(time), "Coins", "Weights", ticks=1,image_name='time2_eigen_portfolio_1.png')

    weights_t_E2 = Eigen_Portfolio_2[Eigen_Portfolio_2.index == time].transpose()
    weights_t_E2 = weights_t_E2[weights_t_E2 != 0].dropna()
    weights_t_E2 = weights_t_E2.squeeze().sort_values(ascending=False)
    fn.plot_graph(weights_t_E2, "Second Largest Eigen Portfolio Weights at " + str(time), "Coins", "Weights", ticks=1, image_name='time2_eigen_portfolio_2.png')

print("Task 2 ended. Largest and Second largest eigen portfolios have been saved")

print("Starting Task 3")
########################################################################################################
# Task 3: Plot the evolution of s-score of BTC and ETH from 2021-09-26 00:00:00 to 2021-10-25 23:00:00
########################################################################################################


T1=datetime(2021,9,26,0)
T2=datetime(2021,10,25,23)

BTC_sscore_sample=BTC_sscore[(BTC_sscore.index>=T1)&(BTC_sscore.index<=T2)]
ETH_sscore_sample=ETH_sscore[(ETH_sscore.index>=T1)&(ETH_sscore.index<=T2)]


fn.plot_graph(BTC_sscore_sample.dropna(),'BTC S-score', 'Time','S-score',0, 'BTC_sscore.png')
fn.plot_graph(ETH_sscore_sample.dropna(),'ETH S-score', 'Time','S-score',0, 'ETH_sscore.png')

print("Task 3 ended. SScore Evolution of BTC and ETH have been saved")
print("Starting Task 4.1")
########################################################################################################
# Task 4.1: Exporting the trading signals & portfolio positions to CSV file
########################################################################################################

Signal_Frame.to_csv('trading_signal.csv')
print("Task 4 ended. Trading Signals have been saved as csv")
print("Starting Task 4.2")
########################################################################################################
# Task 4.2: Calculating and plotting portfolio parameters (cumulative returns,sharpe ratio, maximum drawdown)
########################################################################################################

strategy_parameters=fn.calculate_parameters(Position_Frame, all_coin_data_ret_test, periods)
cumulative_returns_portfolio=strategy_parameters['cumulative_returns']
returns_portfolio=strategy_parameters['portfolio_returns']

fn.plot_graph(cumulative_returns_portfolio,'Portfolio Cumulative Returns', 'Time','Returns',0, 'cumulative_return.png')
#Maximum Drawdown (considering an initial investment of $100)
portfolio_value=(cumulative_returns_portfolio+1)*100
#Sharpe Ratio
sharpe_ratio=strategy_parameters['sharpe']
print('Sharpe Ratio of the strategy is: ',round(sharpe_ratio,2))

# calculating max_value and daily drawdown for window size of 240
max_value = portfolio_value.rolling(periods, min_periods=1).max()
hourly_drawdown = portfolio_value/max_value - 1.0
hourly_drawdown_max = hourly_drawdown.min()
print('Maximum Drawdown at end of the testing period is: ',round(hourly_drawdown_max,4)*100,'%')
ax = returns_portfolio.plot.hist(title='Daily Returns',figsize=(20,7),bins=25)
ax.figure.savefig('hist_return.png')


print("Task 4.2 ended. Portfolio Cummulative Return and histogram of portfolio return has been saved")
