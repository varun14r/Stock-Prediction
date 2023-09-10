# Stock-Prediction
This Python script utilizes Long Short-Term Memory (LSTM) neural networks to predict stock prices. It downloads historical stock data for Apple Inc. (AAPL) from Yahoo Finance, preprocesses the data, trains an LSTM model, and makes predictions. The code includes visualization of historical stock prices, model predictions, and evaluation using Root Mean Squared Error (RMSE).
# Contents:
1. Importing necessary libraries.
2. Setting up the date range and downloading historical stock data.
3. Exploring the data and visualizing the closing price history.
4. Data preprocessing, including scaling and splitting into training and testing sets.
5. Building an LSTM model with two LSTM layers and two dense layers.
6. Compiling the model with an optimizer and loss function.
7. Preparing test data for prediction.
8. Making stock price predictions and calculating RMSE.
9. Visualizing predictions and actual prices.
10. Predicting stock prices for a specific date range.
11. Displaying actual stock prices for the same date range.
Usage: You can use this code as a reference for predicting stock prices for other companies by changing the stock symbol (e.g., 'AAPL' to the desired company's symbol) and adjusting the date range. Ensure you have the required libraries installed.
