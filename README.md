# Stock-prediction-using-Linear-Regression

This repository contains a simple implementation of a stock price prediction model using the linear regression algorithm from scikit-learn. The model is trained and evaluated on historical stock data for Tesla (TSLA) obtained from Yahoo Finance.

The data is first pre-processed by setting the index to the date of the observation, and only the adjusted close price is kept for the analysis. The code then calculates the exponential moving average (EMA) of the close price over a 10-day window, and uses this as the target variable to predict.

The model is trained on 80% of the data and evaluated on the remaining 20% using mean absolute error, coefficient of determination (R^2), and the model coefficients as evaluation metrics.

##Usage
The repository contains the following files:

    TSLA.csv: The stock price data for Tesla obtained from Yahoo Finance
    stock_price_prediction.py: The main script that trains and evaluates the model


To run the model, simply clone the repository and run the stock_price_prediction.py script using your preferred Python environment. The script produces a graph showing the actual and predicted values, and the evaluation metrics.
