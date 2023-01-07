import functions as fn
import pandas as pd
import numpy as np
import generating_trading_signals as gts


class implementation_steps:
    def __init__(self, top40_coin_info, hour_delta, all_coin_data_ret, delta_t, sbo, sso, sbc, ssc, t_start, t_end, window):
        self.top40_coin_info = top40_coin_info
        self.hour_delta = hour_delta
        self.all_coin_data_ret  = all_coin_data_ret
        self.delta_t = delta_t
        self.sbo = sbo
        self.sso = sso
        self.sbc = sbc
        self.ssc = ssc
        self.t_start = t_start
        self.t_end = t_end
        self.window = window

    def strategy_implementation(self, t_rel, position_frame):

        ### Part 1: Computing factor returns of the two risk factors at any given time

        #### Calculating Returns
        comp_function = fn.computation_function(self.delta_t, self.all_coin_data_ret, self.t_start, self.t_end, self.window)
        solution = comp_function.calculate_factor_returns(t_rel, self.top40_coin_info)

        top40_coins_returns = solution['return_matrix']

        #### Calculating Correlations & Eigen Vectors
        correl_matrix = solution['correl_matrix']
        vector_1 = solution['vector1']
        vector_2 = solution['vector2']

        #### Calculating Risk Factors & Factor Returns
        factor_1 = solution['factor1']
        factor_2 = solution['factor2']
        factor_1_returns = solution['factor1_returns']
        factor_2_returns = solution['factor2_returns']

        ### Part 2: Estimating residual returns of given tokens
        #### Calculating Residual Returns
        coin_list = top40_coins_returns.columns

        residual_returns = pd.DataFrame()

        for coin in coin_list:
            Y = top40_coins_returns[coin].reset_index(drop=True)
            X = pd.DataFrame([factor_1_returns, factor_2_returns]).transpose()
            temp = fn.apply_regression(Y, X).resid.rename(coin)

            residual_returns = pd.concat([residual_returns, temp], axis=1)

        #### Calculating Auxillary Process Parameters
        aux_process_data = residual_returns.cumsum()
        delta_t = 1 / 8760

        index_val = ['kappa', 'm', 'sigma', 'sigma_eq', 'b']
        aux_params = pd.DataFrame(index=index_val)


        for coin in aux_process_data.columns:
            regress_fun = fn.computation_function(self.delta_t, self.all_coin_data_ret, self.t_start, self.t_end, self.window)
            coin_residuals_data = aux_process_data[coin]
            coin_params = regress_fun.calculate_process_parameters(coin_residuals_data, index_val, coin)
            aux_params = pd.concat([aux_params, coin_params], axis=1)



        ### Part 3: Generating trading signals at any given time
        s_score = aux_params.transpose()['m'].div(aux_params.transpose()['sigma_eq']).mul(-1)
        s_score = s_score.rename('s_score').to_frame()
        BTC_score = s_score[s_score.index == 'BTC'].values[0]
        ETH_score = s_score[s_score.index == 'ETH'].values[0]
        s_score['b'] = aux_params.transpose()['b']


        position_frame_T = position_frame.transpose()
        position_frame_T_sliced = pd.concat(
            [position_frame_T[t_rel - self.hour_delta].rename('hist_position'), s_score],
            axis=1)


        # signal_generation = gts.tradingSignal(self.sbo,self.sso, self.sbc, self.ssc)
        signals = gts.tradingSignal(self.sbo,self.sso, self.sbc, self.ssc)
        signal_generation = signals.generate_trade_signal(position_frame_T_sliced.copy())

        position_frame_T[t_rel] = signal_generation['new_position']
        position_frame = position_frame_T.transpose()

        return position_frame,signal_generation['signal'],vector_1,vector_2,factor_1,factor_2,BTC_score,ETH_score
