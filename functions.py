import pandas as pd
import numpy as np
from numpy.linalg import eig
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
import math
import warnings


warnings.filterwarnings("ignore")

def plot_graph(data,title,xlab,ylab,ticks, image_name):
    data.plot(figsize=(20,7))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if ticks==1:
        plt.tick_params(axis='x',labelsize=8)
        plt.xticks(ticks=np.arange(1,data.shape[0]+1,1),labels=data.index.to_list())
    plt.savefig(image_name)
    plt.show()


def calculate_parameters(position_frame, all_coin_data_ret_test, periods):
    parameters = {}

    returns = position_frame.shift(1) * all_coin_data_ret_test
    weighted_sum_abs = position_frame.abs().sum(axis=1)
    portfolio_returns = returns.sum(axis=1)
    portfolio_returns = portfolio_returns.div(weighted_sum_abs)


    portfolio_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    portfolio_returns.fillna(0, inplace=True)
    cumulative_returns = (portfolio_returns + 1).cumprod() - 1

    parameters['cumulative_returns'] = cumulative_returns
    parameters['portfolio_returns'] = portfolio_returns
    sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * math.sqrt(periods)

    parameters['sharpe'] = sharpe_ratio

    return parameters

def apply_regression(y, x):
    x = sm.add_constant(x)
    ols = sm.OLS(y, x)
    ols_result = ols.fit()

    return (ols_result)

class computation_function:
    def __init__(self, delta_t, all_coin_data_ret, t_start, t_end, window):
        self.delta_t = delta_t
        self.all_coin_data_ret = all_coin_data_ret
        self.t_start = t_start
        self.t_end = t_end
        self.window = window


    def calculate_factor_returns(self, time, info_frame):

        output = {}
# fetching the return data for the given date & time for top 40 coins at that time (not including the returns of that period)
        top40_coins_returns = self.slice_returns(time, info_frame)
        na_coins = top40_coins_returns.columns[top40_coins_returns.isnull().any()].to_list()
        top40_coins_returns.drop(columns=na_coins, inplace=True)
        top40_coins_returns_stan = self.standardize_data(top40_coins_returns)

        # calculating correlations
        correl_matrix = top40_coins_returns_stan.corr()
        vector1, vector2 = self.apply_pca(correl_matrix)
        vector1 = pd.DataFrame(vector1, index=top40_coins_returns.columns)
        vector2 = pd.DataFrame(vector2, index=top40_coins_returns.columns)

        factor1 = vector1.squeeze().div(top40_coins_returns.std(ddof=1))
        factor2 = vector2.squeeze().div(top40_coins_returns.std(ddof=1))

        # factor1 = vector1 / top40_coins_returns.std().values
        # factor2 = vector2 / top40_coins_returns.std().values

        factor1_returns = np.dot(factor1, top40_coins_returns.transpose())
        factor2_returns = np.dot(factor2, top40_coins_returns.transpose())

        output['return_matrix'] = top40_coins_returns
        output['correl_matrix'] = correl_matrix
        output['vector1'] = vector1
        output['vector2'] = vector2
        output['factor1'] = factor1
        output['factor2'] = factor2
        output['factor1_returns'] = factor1_returns
        output['factor2_returns'] = factor2_returns

        return output


    def slice_returns(self, time, info_frame):
        top40_coins = info_frame[info_frame['startTime'] == time]
        top40_coins = top40_coins.iloc[:, -40:].values.tolist()[0]

        self.all_coin_data_ret.reset_index(inplace=True)
        index_n = self.all_coin_data_ret.index[self.all_coin_data_ret['startTime'] == time].tolist()[0]

        top40_coins_data = self.all_coin_data_ret.iloc[index_n - self.window:index_n, :]
        top40_coins_data.set_index('startTime', inplace=True)
        top40_coins_data = top40_coins_data[top40_coins]

        self.all_coin_data_ret.set_index('startTime', inplace=True)

        return top40_coins_data


    def standardize_data(self, dataset):
        scaled_data = pd.DataFrame(index=dataset.index)
        for col in dataset.columns:
            # using sample mean with ddof (delta degrees of freedom) set to 1
            scaled_data[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std(ddof = 1)
        return scaled_data


    def apply_pca(self, correl_matrix):
        w, v = eig(correl_matrix)
        return v[:, 0], v[:, 1]


    def calculate_process_parameters(self, coin_residuals, index, coin):
        Y = coin_residuals.iloc[1:, ]
        X = coin_residuals.shift(1).iloc[1:, ]
        regression_output = apply_regression(Y, X)

        parameters = regression_output.params
        res_variance = ((regression_output.resid).std()) ** 2

        a = parameters[0]
        b = parameters[1]
        kappa = -math.log(b) / self.delta_t
        m = a / (1 - b)
        # when b value is too close to 1 the mean-reversion time is too long and the model is rejected for the stock under consideration
        if b >=1:
            sigma = float('nan')
            sigma_eq = float('nan')
        else:
            sigma = math.sqrt((res_variance * 2 * kappa) / (1 - b ** 2))
            sigma_eq = math.sqrt((res_variance) / (1 - b ** 2))

        output = pd.Series(data=[kappa, m, sigma, sigma_eq, b], index=index).rename(coin)

        return output
