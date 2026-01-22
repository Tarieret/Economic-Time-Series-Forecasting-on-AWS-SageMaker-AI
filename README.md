# Economic-Time-Series-Forecasting-on-AWS-SageMaker AI with ARIMA, Prophet, and LSTM

![Prophet Forecast vs Actual CPI](images/LSTM_CPI.png)

For this project, I forecasts U.S. Consumer Price Index (CPI) values using classical statistical models, trend-based forecasting, and deep learning. The goal was  to compare model performance under a proper time-series validation framework and deploy  using AWS SageMaker AI. 

## Data
- Source: Federal Reserve Economic Data (FRED)
- Series: CPIAUCSL
- Frequency: Monthly

## Methods
- ARIMA (baseline statistical model)
- Prophet (trend + seasonality model)
- LSTM (deep learning model)

A 24-month holdout period is used for evaluation.

## Results
The LSTM model achieved the lowest MAE and RMSE on the holdout set, Outperforming both ARIMA and Prophet:

**Performance on Holdout Set (Dec 2023 – Nov 2025):**

| Model             | MAE   | RMSE  |
|------------------|-------|-------|
| LSTM (lookback=24) | 3.29  | 3.37  |
| Prophet           | 7.78  | 7.82  |
| ARIMA (1,1,1)     | 8.35  | 9.59  |


## *Deployment
The selected model is serialized and packaged into a `model.tar.gz`.  I also included a custom inference handler and optional deployment script and I intentionally excluded deployment from the notebook to avoid unnecessary cloud costs.
