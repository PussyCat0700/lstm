# List Of Models
## LSTM
Short-Term Wind Power Forecast Based on Continuous Conditional Random Field
## FFNN
AWNN-Assisted Wind Power Forecasting Using Feed-Forward Neural Network
## GPCF
A High-Accuracy Wind Power Forecasting Model (Letter)
## XGBOOST
濮博【Adapted，原先功率他做的是分位预测，我把他竞价预测的拟合方法直接挪到功率上用了】
## GPNN
Xin Yu's Implementation
# Steps
1. create a yaml configuration file under `./conf`.
1. Use `preprocess.py`
1. Train! But please note that checkpoint path for several models need to be manually specified.
1. Gather your results with `gather_results.py`