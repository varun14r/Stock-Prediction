# Stock-Prediction
This Python script utilizes Long Short-Term Memory (LSTM) neural networks to predict stock prices. It downloads historical stock data for Apple Inc. (AAPL) from Yahoo Finance, preprocesses the data, trains an LSTM model, and makes predictions. The code includes visualization of historical stock prices, model predictions, and evaluation using Root Mean Squared Error (RMSE).

## Table of Contents
- Prerequisites
- Usage
- Project Structure
- Results
## Prerequisites
To run the code and experiments, you'll need the following:

1. Python (version 3.6 or later)
2. Jupyter Notebook (to interact with the provided notebook)
3. TensorFlow (for deep learning operations)
4. Keras (for building and training the neural network)
5. NumPy (for array operations)
6. Matplotlib (for visualization)
7. pandas(Data manipulation and analysis)
8. yfinance(Fetch historical stock price data from Yahoo Finance)

## Usage
  
1. You can install the required libraries using the following command:

		pip install tensorflow keras yfinance pandas numpy matplotlib

2. Clone the repository to your local machine:
 
		https://github.com/varun14r/Stock-Prediction.git
   
4. Navigate to the project directory:

		cd Stock-Prediction
5. Run the Jupyter Notebook file:

		jupyter notebook Stock-Prediction.ipynb

## Results

1. The model's performance is evaluated using metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) score.

 	![image](https://github.com/varun14r/Stock-Prediction/assets/18166488/37202e92-7beb-4529-b65d-9c085cd87c0d)

3. Actual vs. predicted stock prices are visualized using Matplotlib.
	
	![image](https://github.com/varun14r/Stock-Prediction/assets/18166488/6ce5f07d-771e-44b8-b329-df3d4df466df)

 
4. You can predict future stock prices for a specific date range using the trained model.

 	![image](https://github.com/varun14r/Stock-Prediction/assets/18166488/a6256700-140a-4ce0-b780-37221acfcbe3)



