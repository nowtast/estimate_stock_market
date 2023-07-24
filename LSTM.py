import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


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


# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # out = self.fc(out[:, -1, :])
        out = self.fc(out)

        return out


class AIModel:
    def __init__(self):
        # Define your model, criterion, optimizer
        self.seq_len = 120
        self.input_size = 12
        self.output_size = 1
        self.hidden_size = 32
        self.num_layers = 3

        self.model = LSTM(self.input_size, self.hidden_size, self.num_layers, self.output_size)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    @staticmethod
    def train_model(features_train, targets_train, model, criterion, optimizer, num_epochs=150):
        for epoch in range(num_epochs):
            inputs = features_train
            labels = targets_train

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(0)[-1], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 150 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    @staticmethod
    def validate_model(features_validation, targets_validation, model, criterion):
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Do not calculate gradients
            input = features_validation
            labels = targets_validation

            outputs = model(input)
            loss = criterion(outputs.squeeze(0)[-1], labels)

        return outputs.squeeze(0), loss.item()

    def gate_train(self, df):
        features = df.iloc[:, :-1].values
        targets = df.iloc[:-1, -1].values

        target_30th_day = df.iloc[-1, -1]

        # Extend targets to 30 days with a default value (e.g., 0)
        targets = np.append(targets, 0)  # 0 is used here, adjust if necessary        # Combine features and targets
        combined_input = np.concatenate([features, targets.reshape(-1, 1)], axis=1)  # Add targets as a new feature

        combined_input = torch.from_numpy(combined_input).float().unsqueeze(0)  # Convert to tensor and add batch dimension
        target_30th_day = torch.tensor(target_30th_day).float().unsqueeze(0)  # Convert to tensor and add batch dimension

        self.train_model(combined_input, target_30th_day, self.model, self.criterion, self.optimizer)

    def gate_validation(self, df):
        features = df.iloc[:, :-1].values
        targets = df.iloc[:-1, -1].values

        target_30th_day = df.iloc[-1, -1]

        # Extend targets to 30 days with a default value (e.g., 0)
        targets = np.append(targets, 0)  # 0 is used here, adjust if necessary        # Combine features and targets
        combined_input = np.concatenate([features, targets.reshape(-1, 1)], axis=1)  # Add targets as a new feature

        combined_input = torch.from_numpy(combined_input).float().unsqueeze(0)  # Convert to tensor and add batch dimension
        target_30th_day = torch.tensor(target_30th_day).float().unsqueeze(0)  # Convert to tensor and add batch dimension

        _result, _loss = self.validate_model(combined_input, target_30th_day, self.model, self.criterion)

        return _result, _loss


if __name__ == "__main__":
    pre_pr = Preprocessor()
    mkmodel = AIModel()
    train_files, val_files = train_test_split(pre_pr.normalized_dataframes, test_size=0.2, random_state=42)
    for normalized_df in train_files:
        mkmodel.gate_train(normalized_df)

    for normalized_df in val_files:
        result, _ = mkmodel.gate_validation(normalized_df)
        print(f'Original: {normalized_df["Invest"][-1]}, Predict: {result}')

    print(mkmodel.model)
    print("end")