import pandas as pd
import numpy as np

class tradingSignal:
    def __init__(self, sbo, sso, sbc, ssc):
        self.sbo = sbo
        self.sso = sso
        self.sbc = sbc
        self.ssc  = ssc

    def generate_trade_signal(self, generator_frame):

        #
        #  Trading rules:
        #  buy to open if si <- sbo
        #  sell to open if si > sso
        #  close short position if si < sbc
        #  close long position if si > -ssc
        # Trading Signals
        #
        # A) +1 when
        #     - As per the given rules
        #         i)  Long position can be taken (provided it does not exist)
        #         ii) Short position needs to be closed (provided it exists)
        #     - Or stock is no longer in top 40 and short position exists
        #
        #
        #
        # B) -1 when
        #     - As per the given rules
        #         i) Short position can be taken (provided it does not exist)
        #         ii) Long position needs to be closed (provided it exists)
        #     - Or stock is no longer in top 40 and long position exists
        #
        #

        # Buy when a short position already exists and the coin is not anymore in the top40
        # (stricter b<1 condition is there to address the situation where s_score is nan because b value was out of bounds)
        condition1 = ((generator_frame['hist_position'] == -1) & (pd.isnull(generator_frame['s_score'])) & (
                    generator_frame['b'] < 1))

        # Buy when a short position already exists and needs to be closed
        condition2 = ((generator_frame['hist_position'] == -1) & (generator_frame['s_score'] < self.sbc) & (
                    generator_frame['b'] < .9672))

        # Buy when a long position can be taken and the coin is not already in the portfolio
        condition3 = ((generator_frame['hist_position'] == 0) & (generator_frame['s_score'] < -self.sbo) & (
                    generator_frame['b'] < .9672))

        # Sell when a long position already exists and the coin is not anymore in the top40
        # (stricter b<1 condition is there to address the situation where s_score is nan because b value was out of bounds)
        condition4 = ((generator_frame['hist_position'] == 1) & (pd.isnull(generator_frame['s_score'])) & (
                    generator_frame['b'] < 1))

        # Sell when a long position already exists and needs to be closed
        condition5 = ((generator_frame['hist_position'] == 1) & (generator_frame['s_score'] > -self.ssc) & (
                    generator_frame['b'] < 9672))

        # Sell when a short position can be taken and the coin is not already shorted in the portfolio
        condition6 = ((generator_frame['hist_position'] == 0) & (generator_frame['s_score'] > self.sso) & (
                    generator_frame['b'] < .9672))

        condition_list = [condition1, condition2, condition3, condition4, condition5, condition6]

        generator_frame['signal'] = np.select(condlist=condition_list, choicelist=[1, 1, 1, -1, -1, -1], default=0)
        generator_frame['new_position'] = generator_frame['hist_position'] + generator_frame['signal']

        return generator_frame

