import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self):
        self.days = 30
        self.df = self.get_stock_data()
        self._get_invest_points()
        self._get_moving_averages()
        self.dataframes = []
        self.normalized_dataframes = []
        self._get_sliced_data()
        self._get_sliced_normalized_data()
        self.num_of_train_data = None
        self.num_of_validation_data = None

    @staticmethod
    def get_stock_data():
        df = fdr.DataReader("005930", start="1975-06-11", end="2023-07-04")
        return df

    @staticmethod
    def normalize_data(row, lv_, hv_):
        return (row - lv_) / (hv_ - lv_)

    def _get_invest_points(self):
        # Calculate the percentage change in closing price
        self.df["Close_change"] = self.df["Close"].pct_change() * 100

        # Set "Invest" column to True for rows where the closing price is higher than the previous day by more than 4%
        self.df["Invest"] = self.df["Close_change"].apply(lambda x: 1 if x > 4 else 0)

        # Shift the "Invest" column by one day to record the decision for the day before
        self.df["Invest"] = self.df["Invest"].shift(-1)
        self.df = self.df.fillna(0)
        # self.df["Invest"].iloc[-1] = 0

    def _get_moving_averages(self):
        # Calculate moving averages
        self.df["MA5"] = self.df["Close"].rolling(window=5).mean()
        self.df["MA20"] = self.df["Close"].rolling(window=20).mean()
        self.df["MA60"] = self.df["Close"].rolling(window=60).mean()
        self.df["MA120"] = self.df["Close"].rolling(window=120).mean()

    def drawing_charts(self):
        # Calculate the number of n-day intervals
        num_intervals = len(self.df) // self.days + 1

        for i in range(4, num_intervals):
            start_idx = i * self.days
            end_idx = min((i + 1) * self.days, len(self.df))
            data_subset = self.df.iloc[start_idx:end_idx]

            # Create a new DataFrame with only the "Invest" column
            invest_data = data_subset[["Invest"]]

            # Plot candlestick chart with moving averages
            apds = [
                mpf.make_addplot(data_subset["MA5"]),
                mpf.make_addplot(data_subset["MA20"]),
                mpf.make_addplot(data_subset["MA60"]),
                mpf.make_addplot(data_subset["MA120"]),
                mpf.make_addplot(invest_data, panel=2, secondary_y=False, color="g")
            ]

            mpf.plot(data_subset, type="candle", addplot=apds, volume=True, style="yahoo", title="Stock Prices")

            plt.show()

    def _get_sliced_data(self):
        # Calculate the number of n-day intervals
        num_intervals = len(self.df) - (self.days - 1)

        # Iterate over the intervals and create the 30-day dataframes
        for i in range(num_intervals):
            start_idx = i
            end_idx = i + self.days
            subset = self.df.iloc[start_idx:end_idx]
            self.dataframes.append(subset)

    def _get_sliced_normalized_data(self):
        # Calculate the number of 30-day intervals
        num_intervals = len(self.df) - (self.days - 1)

        # Iterate over the intervals and create the 30-day normalized dataframes
        scaling_columns = ["MA5", "MA20", "MA60", "MA120", "Open", "High", "Low", "Close"]
        for i in range(120, num_intervals):
            start_idx = i
            end_idx = i + self.days
            subset = self.df.iloc[start_idx:end_idx]

            # Get the highest and lowest values for scaling
            highest_value = subset[scaling_columns].max().max()
            lowest_value = subset[scaling_columns].min().min()

            # Normalize the subset dataframe
            normalized_subset = subset[scaling_columns].apply(self.normalize_data, axis=1, lv_=lowest_value,
                                                              hv_=highest_value)

            normalized_subset['Change'] = subset['Change']
            normalized_subset['Close_change'] = subset['Close_change']

            scaler_volume = MinMaxScaler()
            normalized_subset_volume = pd.DataFrame(scaler_volume.fit_transform(subset),
                                                    columns=subset.columns,
                                                    index=subset.index)
            normalized_subset['Volume'] = normalized_subset_volume['Volume']
            normalized_subset['Invest'] = subset['Invest']
            self.normalized_dataframes.append(normalized_subset)


class GBMModel:
    def __init__(self, df_full, df_sliced):
        self.df_full = df_full
        self.df_sliced = df_sliced

    @staticmethod
    def calc(df):
        df['Return'] = np.log(df['Close'].copy()).diff()

        # Estimate mu (drift) and sigma (volatility)
        mu = df['Return'].mean()
        sigma = df['Return'].std(ddof=1)

        # The last price
        S0 = df['Close'].iloc[-1]

        # Define the time increment (1 day)
        dt = 1 / 252  # there are typically 252 trading days in a year

        # Number of simulations
        num_simulations = 1000

        # Initialize an empty list to store the simulation results
        simulations = []

        for i in range(num_simulations):
            # Generate a random number
            rand_num = np.random.normal(0, 1)

            # Simulate the price for the next day using the GBM formula
            next_day_price = S0 * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand_num)

            # Store the simulation result
            simulations.append(next_day_price)

        # Compute the average of the simulations
        predicted_price = np.mean(simulations)

        print(f"Today's price: {S0}")
        print('Predicted price for the next day (Monte Carlo): ', predicted_price)


if __name__ == "__main__":
    pre_pr = Preprocessor()
    model = GBMModel(pre_pr.df, pre_pr.dataframes[-1])
    model.calc(model.df_full)
    model.calc(model.df_sliced)