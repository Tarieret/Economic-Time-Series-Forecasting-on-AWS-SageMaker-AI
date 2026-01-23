# Economic-Time-Series-Forecasting-on-AWS-SageMaker AI with ARIMA, Prophet, and LSTM

## 📊 Model Comparison Results
![Model Comparison](Images/RMSE_Comparison.png)

## 🏆 Champion Model: LSTM Performance
![LSTM Performance](Images/LSTM_CPI.png)

## **📈 Project Overview**
This project forecasts U.S. Consumer Price Index (CPI) values using a multi-model approach, ranging from classical statistical methods to advanced deep learning. The primary objective was to evaluate model accuracy under a rigorous time-series validation framework and demonstrate production-level deployment.

## Key Highlights

- Model Comparison: Evaluated performance across classical statistical models, trend-based forecasting, and Long Short-Term Memory (LSTM) networks.

- Performance: The LSTM model significantly outperformed statistical baselines. By utilizing a 12-month lookback window and a specialized neural architecture, it captured complex inflationary patterns that traditional models failed to detect.

- Deployment: Fully integrated and deployed using AWS SageMaker AI for scalable inference.



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
The LSTM model achieved the lowest MAE and RMSE on the holdout set by far, Outperforming both ARIMA and Prophet:

**Performance on Holdout Set (Dec 2023 – Nov 2025):**

| Model             | MAE   | RMSE  |
|------------------|-------|-------|
| LSTM (lookback=24) | 0.53  | 0.62  |
| Prophet           | 7.09  | 7.15  |
| ARIMA (1,1,1)     | 8.04  | 9.29  |

## Technical Challenges & Hyperparameter Tuning

Achieving a champion RMSE of 0.619 required navigating several key trade-offs during the model selection process:

- The Lookback Window: Initial 3-month windows missed broader economic cycles, while 24-month windows introduced "gradient noise." A 12-month lookback proved optimal, balancing annual seasonality with responsiveness to recent shifts.

- Model Complexity: Deep architectures (3+ layers) led to vanishing gradients and overfitting on the sparse monthly CPI data. I pivoted to a single-layer LSTM with 50 units, prioritizing a lean design that generalized better on holdout data.

- Optimization Stability: A standard learning rate (0.001) caused erratic validation loss. I stabilized convergence by implementing a Learning Rate Scheduler and reducing the step size to handle the non-stationary nature of inflation data.

## Deployment
The selected model is serialized and packaged into a `model.tar.gz`.  I also included a custom inference handler and optional deployment script and I intentionally excluded deployment from the notebook to avoid unnecessary cloud costs.
