import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import FinanceDataReader as fdr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler


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
        df = fdr.DataReader("005930", start="1975-06-11", end="2023-07-03")
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

class AIModel:
    def __init__(self):
        # Create an SVM classifier
        self.clf = svm.SVC(kernel='rbf', gamma='scale', C=1.0)

    def gate_train(self, x_train, y_train):
        # Scale the features

        if np.unique(y_train).shape[0] < 2:
            print(f"Skipping this data set as it has only one class.")
        else:
            # Train the classifier
            self.clf.fit(x_train, y_train)


    def gate_validation(self, x_test, y_test):
        # After training with all data files, you can test the classifier's performance:
        accuracy = self.clf.score(x_test, y_test)
        print('Model Accuracy: ', accuracy)

        # Make predictions
        y_pred = self.clf.predict(x_test)
        # Print predicted labels
        print(f'Original labels: {y_test}, Predicted labels: {y_pred}')


if __name__ == "__main__":
    pre_pr = Preprocessor()
    mkmodel = AIModel()
    train_files, val_files = train_test_split(pre_pr.normalized_dataframes, test_size=0.2, random_state=42)
    for normalized_df in train_files:
        X = (normalized_df.drop(['Invest'], axis=1)).to_numpy()
        y = normalized_df['Invest'].values

        mkmodel.gate_train(X, y)

    for normalized_df in val_files:
        X = (normalized_df.drop(['Invest'], axis=1)).to_numpy()
        y = normalized_df['Invest'].values

        mkmodel.gate_validation(X, y)

    print(mkmodel.model)
    print("end")
